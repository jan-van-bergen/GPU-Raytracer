#include <vector_types.h>
#include <corecrt_math.h>

#include "cuda_math.h"

#include "../Common.h"

#include "Tracing.h"
#include "Lighting.h"
#include "Sky.h"
#include "Sampler.h"
#include "Util.h"

#define sigma_z 1.0f
#define sigma_n 128.0f
#define sigma_l 400.0f

#define epsilon 1e-8f // To avoid division by 0

// Frame Buffers
__device__ float4 * frame_buffer_albedo;
__device__ float4 * frame_buffer_moment;

__device__ float4 * frame_buffer_direct;
__device__ float4 * frame_buffer_indirect;

surface<void, 2> accumulator; // Final Frame buffer to be displayed on Screen

// GBuffers
texture<float4, cudaTextureType2D> gbuffer_normal_and_depth;
texture<float2, cudaTextureType2D> gbuffer_uv;
texture<int,    cudaTextureType2D> gbuffer_triangle_id;
texture<float2, cudaTextureType2D> gbuffer_motion;
texture<float2, cudaTextureType2D> gbuffer_depth_gradient;

// History Buffers (Temporally Integrated)
__device__ int    * history_length;
__device__ float4 * history_direct;
__device__ float4 * history_indirect;
__device__ float4 * history_moment;
__device__ float4 * history_normal_and_depth;
__device__ int    * history_triangle_id;

__device__ void frame_buffer_add(float4 * frame_buffer, int x, int y, const float3 & colour) {
	ASSERT(x >= 0 && x < SCREEN_WIDTH);
	ASSERT(y >= 0 && y < SCREEN_HEIGHT);

	frame_buffer[x + y * SCREEN_WIDTH] += make_float4(colour, 0.0f);
}

// Vector3 in AoS layout
struct Vector3 {
	float * x;
	float * y;
	float * z;

	__device__ void from_float3(int index, const float3 & vector) {
		x[index] = vector.x;
		y[index] = vector.y;
		z[index] = vector.z;
	}

	__device__ float3 to_float3(int index) const {
		return make_float3(
			x[index],
			y[index],
			z[index]
		);
	}
};

// Input to the Extend Kernel in SoA layout
struct ExtendBuffer {
	// Ray related
	Vector3 origin;
	Vector3 direction;

	// Pixel colour related
	int   * pixel_index;
	Vector3 throughput;

	// Material related
	char  * last_material_type;
	float * last_pdf;
};

// Input to the various Shade Kernels in SoA layout
struct MaterialBuffer {
	// Ray related
	Vector3 direction;	
	
	// Hit related
	int   * triangle_id;
	float * u;
	float * v;

	// Pixel colour related
	int   * pixel_index;
	Vector3 throughput;
};

// Input to the Connect Kernel in SoA layout
struct ShadowRayBuffer {
	// Ray related
	Vector3 prev_direction_in;

	// Hit related
	int   * triangle_id;
	float * u;
	float * v;

	// Pixel colour related
	int   * pixel_index;
	Vector3 throughput;
};

__device__ ExtendBuffer    ray_buffer_extend;
__device__ MaterialBuffer  ray_buffer_shade_diffuse;
__device__ MaterialBuffer  ray_buffer_shade_dielectric;
__device__ MaterialBuffer  ray_buffer_shade_glossy;
__device__ ShadowRayBuffer ray_buffer_connect;

// Number of elements in each Buffer
struct BufferSizes {
	int N_extend    [NUM_BOUNCES];
	int N_diffuse   [NUM_BOUNCES];
	int N_dielectric[NUM_BOUNCES];
	int N_glossy    [NUM_BOUNCES];
	int N_shadow    [NUM_BOUNCES];
};

__device__ BufferSizes buffer_sizes;

// Sends the rasterized GBuffer to the right Material kernels,
// as if the primary Rays they were Raytraced 
extern "C" __global__ void kernel_primary(
	float3 camera_position,
	float3 camera_bottom_left_corner,
	float3 camera_x_axis,
	float3 camera_y_axis
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float u_screenspace = float(x) + 0.5f;
	float v_screenspace = float(y) + 0.5f;

	float u = u_screenspace / float(SCREEN_WIDTH);
	float v = v_screenspace / float(SCREEN_HEIGHT);

	float2 uv          = tex2D(gbuffer_uv,          u, v);
	int    triangle_id = tex2D(gbuffer_triangle_id, u, v) - 1;

	float3 ray_direction = normalize(camera_bottom_left_corner
		+ u_screenspace * camera_x_axis
		+ v_screenspace * camera_y_axis
	);

	// Triangle ID -1 means no hit
	if (triangle_id == -1) {
		frame_buffer_add(frame_buffer_albedo, x, y, sample_sky(ray_direction));
		frame_buffer_add(frame_buffer_direct, x, y, make_float3(1.0f, 1.0f, 1.0f));

		return;
	}

	const Material & material = materials[triangles_material_id[triangle_id]];

	if (material.type == Material::Type::LIGHT) {
		frame_buffer_add(frame_buffer_albedo, x, y, material.emission);
		frame_buffer_add(frame_buffer_direct, x, y, make_float3(1.0f, 1.0f, 1.0f));
	} else if (material.type == Material::Type::DIFFUSE) {
		int index_out = atomic_agg_inc(&buffer_sizes.N_diffuse[0]);

		ray_buffer_shade_diffuse.triangle_id[index_out] = triangle_id;
		ray_buffer_shade_diffuse.u[index_out] = uv.x;
		ray_buffer_shade_diffuse.v[index_out] = uv.y;

		ray_buffer_shade_diffuse.pixel_index[index_out] = pixel_index;
		ray_buffer_shade_diffuse.throughput.from_float3(index_out, make_float3(1.0f, 1.0f, 1.0f));
	} else if (material.type == Material::Type::DIELECTRIC) {
		int index_out = atomic_agg_inc(&buffer_sizes.N_dielectric[0]);

		ray_buffer_shade_dielectric.direction.from_float3(index_out, ray_direction);

		ray_buffer_shade_dielectric.triangle_id[index_out] = triangle_id;
		ray_buffer_shade_dielectric.u[index_out] = uv.x;
		ray_buffer_shade_dielectric.v[index_out] = uv.y;

		ray_buffer_shade_dielectric.pixel_index[index_out] = pixel_index;
		ray_buffer_shade_dielectric.throughput.from_float3(index_out, make_float3(1.0f, 1.0f, 1.0f));
	} else if (material.type == Material::Type::GLOSSY) {
		int index_out = atomic_agg_inc(&buffer_sizes.N_glossy[0]);

		ray_buffer_shade_glossy.direction.from_float3(index_out, ray_direction);

		ray_buffer_shade_glossy.triangle_id[index_out] = triangle_id;
		ray_buffer_shade_glossy.u[index_out] = uv.x;
		ray_buffer_shade_glossy.v[index_out] = uv.y;

		ray_buffer_shade_glossy.pixel_index[index_out] = pixel_index;
		ray_buffer_shade_glossy.throughput.from_float3(index_out, make_float3(1.0f, 1.0f, 1.0f));
	}
}

extern "C" __global__ void kernel_generate(
	int rand_seed,
	int sample_index,
	float3 camera_position,
	float3 camera_bottom_left_corner,
	float3 camera_x_axis,
	float3 camera_y_axis
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= SCREEN_WIDTH * SCREEN_HEIGHT) return;

	unsigned seed = (index + rand_seed * 199494991) * 949525949;
	
	int block_index = index / BLOCK_SIZE;
	int i = (block_index % (SCREEN_WIDTH / BLOCK_WIDTH)) * BLOCK_WIDTH;
	int j = (block_index / (SCREEN_WIDTH / BLOCK_WIDTH)) * BLOCK_HEIGHT;

	ASSERT(i < SCREEN_WIDTH, "");
	ASSERT(j < SCREEN_HEIGHT, "");

	int k = (index % BLOCK_SIZE) % BLOCK_WIDTH;
	int l = (index % BLOCK_SIZE) / BLOCK_WIDTH;

	ASSERT(k < BLOCK_WIDTH, "");
	ASSERT(l < BLOCK_HEIGHT, "");

	int x = i + k;
	int y = j + l;

	ASSERT(x < SCREEN_WIDTH, "");
	ASSERT(y < SCREEN_HEIGHT, "");

	int pixel_index = x + y * SCREEN_WIDTH;
	ASSERT(pixel_index < SCREEN_WIDTH * SCREEN_HEIGHT, "Pixel should be on screen");

	// Add random value between 0 and 1 so that after averaging we get anti-aliasing
	float u = float(x) + random_float_heitz(x, y, sample_index, 0, 0, seed);
	float v = float(y) + random_float_heitz(x, y, sample_index, 0, 1, seed);
	
	// Create primary Ray that starts at the Camera's position and goes through the current pixel
	ray_buffer_extend.origin.from_float3(index, camera_position);
	ray_buffer_extend.direction.from_float3(index, normalize(camera_bottom_left_corner
		+ u * camera_x_axis
		+ v * camera_y_axis
	));

	ray_buffer_extend.pixel_index[index]  = pixel_index;
	ray_buffer_extend.throughput.x[index] = 1.0f;
	ray_buffer_extend.throughput.y[index] = 1.0f;
	ray_buffer_extend.throughput.z[index] = 1.0f;

	ray_buffer_extend.last_material_type[index] = char(Material::Type::DIELECTRIC);
}

extern "C" __global__ void kernel_extend(int rand_seed, int bounce) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.N_extend[bounce]) return;

	float3 ray_origin    = ray_buffer_extend.origin.to_float3(index);
	float3 ray_direction = ray_buffer_extend.direction.to_float3(index);

	Ray ray;
	ray.origin    = ray_origin;
	ray.direction = ray_direction;
	ray.direction_inv = make_float3(
		1.0f / ray.direction.x, 
		1.0f / ray.direction.y, 
		1.0f / ray.direction.z
	);

	RayHit hit;
	mbvh_trace(ray, hit);

	int ray_pixel_index = ray_buffer_extend.pixel_index[index];
	int x = ray_pixel_index % SCREEN_WIDTH;
	int y = ray_pixel_index / SCREEN_WIDTH; 

	// If we didn't hit anything, sample the Sky
	if (hit.t == INFINITY) {
		float3 illumination = ray_buffer_extend.throughput.to_float3(index) * sample_sky(ray_direction);

		if (bounce == 1) {
			frame_buffer_add(frame_buffer_direct, x, y, illumination);
		} else {
			frame_buffer_add(frame_buffer_indirect, x, y, illumination);
		}

		return;
	}

	float3 ray_throughput = ray_buffer_extend.throughput.to_float3(index);

	unsigned seed = (index + rand_seed * 906313609) * 341828143;

	// Russian Roulette termination
	float p_survive = clamp(max(ray_throughput.x, max(ray_throughput.y, ray_throughput.z)), 0.0f, 1.0f);
	if (random_float_xorshift(seed) > p_survive) {
		return;
	}

	ray_throughput /= p_survive;

	// Get the Material of the Triangle we hit
	const Material & material = materials[triangles_material_id[hit.triangle_id]];

	if (material.type == Material::Type::LIGHT) {
		int x = ray_pixel_index % SCREEN_WIDTH;
		int y = ray_pixel_index / SCREEN_WIDTH; 

		if ((ray_buffer_extend.last_material_type[index] == char(Material::Type::DIELECTRIC)) ||
			(ray_buffer_extend.last_material_type[index] == char(Material::Type::GLOSSY) && material.roughness < ROUGHNESS_CUTOFF)) {
			float3 illumination = ray_throughput * material.emission;
			
			frame_buffer_add(frame_buffer_indirect, x, y, illumination);

			if (bounce == 1) {
				frame_buffer_add(frame_buffer_direct, x, y, illumination);
			} else {
				frame_buffer_add(frame_buffer_indirect, x, y, illumination);
			}

			return;
		}

		float3 light_point  = barycentric(hit.u, hit.v, triangles_position0[hit.triangle_id], triangles_position_edge1[hit.triangle_id], triangles_position_edge2[hit.triangle_id]);
		float3 light_normal = barycentric(hit.u, hit.v, triangles_normal0[hit.triangle_id],   triangles_normal_edge1[hit.triangle_id],   triangles_normal_edge2[hit.triangle_id]);
	
		light_normal = normalize(light_normal);
	
		float3 to_light = light_point - ray_origin;;
		float distance_to_light_squared = dot(to_light, to_light);
		float distance_to_light         = sqrtf(distance_to_light_squared);
	
		// Normalize the vector to the light
		to_light /= distance_to_light;
		
		float cos_o = -dot(to_light, light_normal);

		float light_area = 0.5f * length(cross(
			triangles_position_edge1[hit.triangle_id], 
			triangles_position_edge2[hit.triangle_id]
		));

		float light_pdf = distance_to_light_squared / (cos_o * light_area); // 1 / solid_angle
		float brdf_pdf  = ray_buffer_extend.last_pdf[index];

		float mis_pdf = light_pdf + brdf_pdf;

		float3 illumination = ray_throughput * material.emission / mis_pdf;

		if (bounce == 1) {
			frame_buffer_add(frame_buffer_direct, x, y, illumination);
		} else {
			frame_buffer_add(frame_buffer_indirect, x, y, illumination);
		}
	} else if (material.type == Material::Type::DIFFUSE) {
		int index_out = atomic_agg_inc(&buffer_sizes.N_diffuse[bounce]);

		ray_buffer_shade_diffuse.triangle_id[index_out] = hit.triangle_id;
		ray_buffer_shade_diffuse.u[index_out] = hit.u;
		ray_buffer_shade_diffuse.v[index_out] = hit.v;

		ray_buffer_shade_diffuse.pixel_index[index_out] = ray_buffer_extend.pixel_index[index];
		ray_buffer_shade_diffuse.throughput.from_float3(index_out, ray_throughput);
	} else if (material.type == Material::Type::DIELECTRIC) {
		int index_out = atomic_agg_inc(&buffer_sizes.N_dielectric[bounce]);

		ray_buffer_shade_dielectric.direction.from_float3(index_out, ray_direction);

		ray_buffer_shade_dielectric.triangle_id[index_out] = hit.triangle_id;
		ray_buffer_shade_dielectric.u[index_out] = hit.u;
		ray_buffer_shade_dielectric.v[index_out] = hit.v;

		ray_buffer_shade_dielectric.pixel_index[index_out] = ray_buffer_extend.pixel_index[index];
		ray_buffer_shade_dielectric.throughput.from_float3(index_out, ray_throughput);
	} else if (material.type == Material::Type::GLOSSY) {
		int index_out = atomic_agg_inc(&buffer_sizes.N_glossy[bounce]);

		ray_buffer_shade_glossy.direction.from_float3(index_out, ray_direction);

		ray_buffer_shade_glossy.triangle_id[index_out] = hit.triangle_id;
		ray_buffer_shade_glossy.u[index_out] = hit.u;
		ray_buffer_shade_glossy.v[index_out] = hit.v;

		ray_buffer_shade_glossy.pixel_index[index_out] = ray_buffer_extend.pixel_index[index];
		ray_buffer_shade_glossy.throughput.from_float3(index_out, ray_throughput);
	}
}

extern "C" __global__ void kernel_shade_diffuse(int rand_seed, int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.N_diffuse[bounce]) return;

	// float3 ray_direction = ray_buffer_shade_diffuse.direction[index];

	int   ray_triangle_id = ray_buffer_shade_diffuse.triangle_id[index];
	float ray_u = ray_buffer_shade_diffuse.u[index];
	float ray_v = ray_buffer_shade_diffuse.v[index];

	int ray_pixel_index = ray_buffer_shade_diffuse.pixel_index[index];
	int x = ray_pixel_index % SCREEN_WIDTH;
	int y = ray_pixel_index / SCREEN_WIDTH; 

	float3 ray_throughput = ray_buffer_shade_diffuse.throughput.to_float3(index);

	ASSERT(ray_triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = (index + rand_seed * 794454497) * 781939187;

	const Material & material = materials[triangles_material_id[ray_triangle_id]];

	ASSERT(material.type == Material::Type::DIFFUSE, "Material should be diffuse in this Kernel");

	float3 hit_point     = barycentric(ray_u, ray_v, triangles_position0 [ray_triangle_id], triangles_position_edge1 [ray_triangle_id], triangles_position_edge2 [ray_triangle_id]);
	float3 hit_normal    = barycentric(ray_u, ray_v, triangles_normal0   [ray_triangle_id], triangles_normal_edge1   [ray_triangle_id], triangles_normal_edge2   [ray_triangle_id]);
	float2 hit_tex_coord = barycentric(ray_u, ray_v, triangles_tex_coord0[ray_triangle_id], triangles_tex_coord_edge1[ray_triangle_id], triangles_tex_coord_edge2[ray_triangle_id]);

	float3 albedo = material.albedo(hit_tex_coord.x, hit_tex_coord.y);
	float3 throughput = ray_throughput;

	if (bounce == 0) {
		frame_buffer_add(frame_buffer_albedo, x, y, albedo);
	} else {
		throughput *= albedo;
	}

	if (light_count > 0) {
		int shadow_ray_index = atomic_agg_inc(&buffer_sizes.N_shadow[bounce]);

		ray_buffer_connect.triangle_id[shadow_ray_index] = ray_triangle_id;
		ray_buffer_connect.u[shadow_ray_index] = ray_u;
		ray_buffer_connect.v[shadow_ray_index] = ray_v;

		ray_buffer_connect.pixel_index[shadow_ray_index] = ray_pixel_index;
		ray_buffer_connect.throughput.from_float3(shadow_ray_index, throughput);
	}

	if (bounce == NUM_BOUNCES - 1) return;

	hit_normal = normalize(hit_normal);
	// if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	int index_out = atomic_agg_inc(&buffer_sizes.N_extend[bounce + 1]);

	float3 direction = random_cosine_weighted_diffuse_reflection(x, y, sample_index, bounce, seed, hit_normal);

	ray_buffer_extend.origin.from_float3(index_out, hit_point);
	ray_buffer_extend.direction.from_float3(index_out, direction);

	ray_buffer_extend.pixel_index[index_out]  = ray_pixel_index;
	ray_buffer_extend.throughput.from_float3(index_out, throughput);

	ray_buffer_extend.last_material_type[index_out] = char(Material::Type::DIFFUSE);
	ray_buffer_extend.last_pdf[index_out] = dot(direction, hit_normal) * ONE_OVER_PI;
}

extern "C" __global__ void kernel_shade_dielectric(int rand_seed, int bounce) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.N_dielectric[bounce] || bounce == NUM_BOUNCES - 1) return;

	float3 ray_direction = ray_buffer_shade_dielectric.direction.to_float3(index);

	int   ray_triangle_id = ray_buffer_shade_dielectric.triangle_id[index];
	float ray_u = ray_buffer_shade_dielectric.u[index];
	float ray_v = ray_buffer_shade_dielectric.v[index];

	int ray_pixel_index = ray_buffer_shade_dielectric.pixel_index[index];
	int x = ray_pixel_index % SCREEN_WIDTH;
	int y = ray_pixel_index / SCREEN_WIDTH; 

	float3 ray_throughput = ray_buffer_shade_dielectric.throughput.to_float3(index);

	ASSERT(ray_triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = (index + rand_seed * 758505857) * 364686463;

	const Material & material = materials[triangles_material_id[ray_triangle_id]];

	ASSERT(material.type == Material::Type::DIELECTRIC, "Material should be dielectric in this Kernel");

	float3 hit_point     = barycentric(ray_u, ray_v, triangles_position0 [ray_triangle_id], triangles_position_edge1 [ray_triangle_id], triangles_position_edge2 [ray_triangle_id]);
	float3 hit_normal    = barycentric(ray_u, ray_v, triangles_normal0   [ray_triangle_id], triangles_normal_edge1   [ray_triangle_id], triangles_normal_edge2   [ray_triangle_id]);
	float2 hit_tex_coord = barycentric(ray_u, ray_v, triangles_tex_coord0[ray_triangle_id], triangles_tex_coord_edge1[ray_triangle_id], triangles_tex_coord_edge2[ray_triangle_id]);

	hit_normal = normalize(hit_normal);
	// if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	int index_out = atomic_agg_inc(&buffer_sizes.N_extend[bounce + 1]);

	float3 direction;
	float3 direction_reflected = reflect(ray_direction, hit_normal);

	float3 normal;
	float  cos_theta;

	float n_1;
	float n_2;

	float dir_dot_normal = dot(ray_direction, hit_normal);
	if (dir_dot_normal < 0.0f) { 
		// Entering material		
		n_1 = 1.0f;
		n_2 = material.index_of_refraction;

		normal    =  hit_normal;
		cos_theta = -dir_dot_normal;
	} else { 
		// Leaving material
		n_1 = material.index_of_refraction;
		n_2 = 1.0f;

		normal    = -hit_normal;
		cos_theta =  dir_dot_normal;
	}

	float eta = n_1 / n_2;
	float k = 1.0f - eta*eta * (1.0f - cos_theta*cos_theta);

	if (k < 0.0f) {
		direction = direction_reflected;
	} else {
		float3 direction_refracted = normalize(eta * ray_direction + (eta * cos_theta - sqrtf(k)) * hit_normal);

		// Use Schlick's Approximation
		float r_0 = (n_1 - n_2) / (n_1 + n_2);
		r_0 *= r_0;

		if (n_1 > n_2) {
			cos_theta = -dot(direction_refracted, normal);
		}

		float one_minus_cos         = 1.0f - cos_theta;
		float one_minus_cos_squared = one_minus_cos * one_minus_cos;

		float F_r = r_0 + ((1.0f - r_0) * one_minus_cos_squared) * (one_minus_cos_squared * one_minus_cos);

		if (random_float_xorshift(seed) < F_r) {
			direction = direction_reflected;
		} else {
			direction = direction_refracted;
		}
	}

	if (bounce == 0) {
		frame_buffer_add(frame_buffer_albedo, x, y, make_float3(1.0f, 1.0f, 1.0f));
	}

	ray_buffer_extend.origin.from_float3(index_out, hit_point);
	ray_buffer_extend.direction.from_float3(index_out, direction);

	ray_buffer_extend.pixel_index[index_out] = ray_pixel_index;
	ray_buffer_extend.throughput.from_float3(index_out, ray_throughput);

	ray_buffer_extend.last_material_type[index_out] = char(Material::Type::DIELECTRIC);
}

extern "C" __global__ void kernel_shade_glossy(int rand_seed, int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.N_glossy[bounce]) return;

	float3 direction_in = -ray_buffer_shade_glossy.direction.to_float3(index);

	int   ray_triangle_id = ray_buffer_shade_glossy.triangle_id[index];
	float ray_u = ray_buffer_shade_glossy.u[index];
	float ray_v = ray_buffer_shade_glossy.v[index];

	int ray_pixel_index = ray_buffer_shade_glossy.pixel_index[index];
	int x = ray_pixel_index % SCREEN_WIDTH;
	int y = ray_pixel_index / SCREEN_WIDTH; 

	float3 ray_throughput = ray_buffer_shade_glossy.throughput.to_float3(index);

	ASSERT(ray_triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = (index + rand_seed * 354767453) * 346434643;

	const Material & material = materials[triangles_material_id[ray_triangle_id]];

	ASSERT(material.type == Material::Type::GLOSSY, "Material should be glossy in this Kernel");

	float3 hit_point     = barycentric(ray_u, ray_v, triangles_position0 [ray_triangle_id], triangles_position_edge1 [ray_triangle_id], triangles_position_edge2 [ray_triangle_id]);
	float3 hit_normal    = barycentric(ray_u, ray_v, triangles_normal0   [ray_triangle_id], triangles_normal_edge1   [ray_triangle_id], triangles_normal_edge2   [ray_triangle_id]);
	float2 hit_tex_coord = barycentric(ray_u, ray_v, triangles_tex_coord0[ray_triangle_id], triangles_tex_coord_edge1[ray_triangle_id], triangles_tex_coord_edge2[ray_triangle_id]);

	float3 albedo = material.albedo(hit_tex_coord.x, hit_tex_coord.y);
	float3 throughput = ray_throughput;

	if (bounce == 0) {
		frame_buffer_add(frame_buffer_albedo, x, y, albedo);
	} else {
		throughput *= albedo;
	}

	if (light_count > 0 && material.roughness >= ROUGHNESS_CUTOFF) {
		int shadow_ray_index = atomic_agg_inc(&buffer_sizes.N_shadow[bounce]);

		ray_buffer_connect.prev_direction_in.from_float3(shadow_ray_index, direction_in);

		ray_buffer_connect.triangle_id[shadow_ray_index] = ray_triangle_id;
		ray_buffer_connect.u[shadow_ray_index] = ray_u;
		ray_buffer_connect.v[shadow_ray_index] = ray_v;

		ray_buffer_connect.pixel_index[shadow_ray_index] = ray_pixel_index;
		ray_buffer_connect.throughput.from_float3(shadow_ray_index, throughput);
	}

	if (bounce == NUM_BOUNCES - 1) return;

	hit_normal = normalize(hit_normal);
	if (dot(direction_in, hit_normal) < 0.0f) hit_normal = -hit_normal;

	// Slightly widen the distribution to prevent the weights from becoming too large (see Walter et al. 2007)
	float alpha = (1.2f - 0.2f * sqrt(dot(direction_in, hit_normal))) * material.roughness;
	
	// Sample normal distribution in spherical coordinates
	float theta = atan(sqrt(-alpha * alpha * log(random_float_heitz(x, y, sample_index, bounce, 4, seed) + 1e-8f)));
	float phi   = TWO_PI * random_float_heitz(x, y, sample_index, bounce, 5, seed);

	float sin_theta, cos_theta;
	float sin_phi,   cos_phi;

	sincos(theta, &sin_theta, &cos_theta);
	sincos(phi,   &sin_phi,   &cos_phi);

	// Convert from spherical coordinates to cartesian coordinates
	float3 micro_normal_local = make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

	float3 hit_tangent, hit_binormal;
	orthonormal_basis(hit_normal, hit_tangent, hit_binormal);

	float3 micro_normal_world = local_to_world(micro_normal_local, hit_tangent, hit_binormal, hit_normal);

	float3 direction_out = reflect(-direction_in, micro_normal_world);

	float i_dot_m = dot(direction_in, micro_normal_world);
	float i_dot_n = dot(direction_in,       hit_normal);
	float o_dot_n = dot(direction_out,      hit_normal);
	float m_dot_n = dot(micro_normal_world, hit_normal);

	float D = beckmann_D(m_dot_n, alpha);
	float G = 
		beckmann_G1(i_dot_n, m_dot_n, alpha) * 
		beckmann_G1(o_dot_n, m_dot_n, alpha);
	float weight = abs(i_dot_m) * G / abs(i_dot_n * m_dot_n);

	int index_out = atomic_agg_inc(&buffer_sizes.N_extend[bounce + 1]);

	ray_buffer_extend.origin.from_float3(index_out, hit_point);
	ray_buffer_extend.direction.from_float3(index_out, direction_out);

	ray_buffer_extend.pixel_index[index_out]  = ray_pixel_index;
	ray_buffer_extend.throughput.from_float3(index_out, throughput);

	ray_buffer_extend.last_material_type[index_out] = char(Material::Type::GLOSSY);
	ray_buffer_extend.last_pdf[index_out] = D * m_dot_n / (4.0f * dot(micro_normal_world, direction_in));
}

extern "C" __global__ void kernel_connect(int rand_seed, int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.N_shadow[bounce]) return;

	int   ray_triangle_id = ray_buffer_connect.triangle_id[index];
	float ray_u = ray_buffer_connect.u[index];
	float ray_v = ray_buffer_connect.v[index];

	int ray_pixel_index = ray_buffer_connect.pixel_index[index];
	int x = ray_pixel_index % SCREEN_WIDTH;
	int y = ray_pixel_index / SCREEN_WIDTH; 

	float3 ray_throughput = ray_buffer_connect.throughput.to_float3(index);

	unsigned seed = (index + rand_seed * 390292093) * 162898261;

	// Pick a random light emitting triangle
	int light_triangle_id = light_indices[random_xorshift(seed) % light_count];

	ASSERT(length(materials[triangles_material_id[light_triangle_id]].emission) > 0.0f, "Material was not emissive!\n");

	// Pick a random point on the triangle using random barycentric coordinates
	float u = random_float_heitz(x, y, sample_index, bounce, 6, seed);
	float v = random_float_heitz(x, y, sample_index, bounce, 7, seed);

	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}

	float3 light_point  = barycentric(u, v, triangles_position0[light_triangle_id], triangles_position_edge1[light_triangle_id], triangles_position_edge2[light_triangle_id]);
	float3 light_normal = barycentric(u, v, triangles_normal0[light_triangle_id],   triangles_normal_edge1[light_triangle_id],   triangles_normal_edge2[light_triangle_id]);

	float3 hit_point  = barycentric(ray_u, ray_v, triangles_position0[ray_triangle_id], triangles_position_edge1[ray_triangle_id], triangles_position_edge2[ray_triangle_id]);
	float3 hit_normal = barycentric(ray_u, ray_v, triangles_normal0  [ray_triangle_id], triangles_normal_edge1  [ray_triangle_id], triangles_normal_edge2  [ray_triangle_id]);

	hit_normal   = normalize(hit_normal);
	light_normal = normalize(light_normal);

	float3 to_light = light_point - hit_point;
	float distance_to_light_squared = dot(to_light, to_light);
	float distance_to_light         = sqrtf(distance_to_light_squared);

	// Normalize the vector to the light
	to_light /= distance_to_light;
	
	float cos_o = -dot(to_light, light_normal);
	float cos_i =  dot(to_light,   hit_normal);

	if (cos_o > 0.0f && cos_i > 0.0f) {
		Ray shadow_ray;
		shadow_ray.origin    = hit_point;
		shadow_ray.direction = to_light;
		shadow_ray.direction_inv = make_float3(
			1.0f / shadow_ray.direction.x, 
			1.0f / shadow_ray.direction.y, 
			1.0f / shadow_ray.direction.z
		);

		// Check if the light is obstructed by any other object in the scene
		if (!mbvh_intersect(shadow_ray, distance_to_light - EPSILON)) {
			const Material & hit_material   = materials[triangles_material_id[ray_triangle_id]];
			const Material & light_material = materials[triangles_material_id[light_triangle_id]];

			float brdf;
			float brdf_pdf;

			if (hit_material.type == Material::Type::DIFFUSE) {
				// NOTE: N dot L is included here
				brdf     = cos_i * ONE_OVER_PI;
				brdf_pdf = cos_i * ONE_OVER_PI;
			} else if (hit_material.type == Material::Type::GLOSSY) {			
				float3 prev_direction_in = ray_buffer_connect.prev_direction_in.to_float3(index);

				float3 half_vector = normalize(to_light + prev_direction_in);

				float alpha = (1.2f - 0.2f * sqrt(cos_i)) * hit_material.roughness;
				
				float i_dot_n = dot(prev_direction_in, hit_normal);
				float m_dot_n = dot(half_vector,       hit_normal);

				// Self-shadowing term (using two monodirectional Smith terms)
				float G =
					beckmann_G1(i_dot_n, m_dot_n, alpha) *
					beckmann_G1(cos_i,   m_dot_n, alpha);

				// Normal Distribution Function: samples the likelihood of observing 'halfvector'
				// as a microsurface normal, given the macrosurface normal 'hit_normal'
				float D = beckmann_D(m_dot_n, alpha);

				// NOTE: N dot L is omitted from the denominator here
				brdf     = (G * D) / (4.0f * i_dot_n);
				brdf_pdf = D * m_dot_n / (4.0f * dot(half_vector, prev_direction_in));
			}

			float light_area = 0.5f * length(cross(
				triangles_position_edge1[light_triangle_id], 
				triangles_position_edge2[light_triangle_id]
			));
			float light_pdf = distance_to_light_squared / (cos_o * light_area); // 1 / solid_angle

			float mis_pdf = brdf_pdf + light_pdf;

			float3 illumination = ray_throughput * brdf * light_count * light_material.emission / mis_pdf;

			if (bounce == 0) {
				frame_buffer_add(frame_buffer_direct, x, y, illumination);
			} else {
				frame_buffer_add(frame_buffer_indirect, x, y, illumination);
			}
		}
	}
}

__device__ bool is_tap_consistent(int x, int y, const float3 & normal, float depth) {
	if (x < 0 || x >= SCREEN_WIDTH)  return false;
	if (y < 0 || y >= SCREEN_HEIGHT) return false;

	float4 prev_normal_and_depth = history_normal_and_depth[x + y * SCREEN_WIDTH];
	
	float3 prev_normal = make_float3(prev_normal_and_depth);
	float  prev_depth  = prev_normal_and_depth.w;

	const float threshold_normal = 0.95f;
	const float threshold_depth  = 0.025f * 250.0f; // @HARDCODED @ROBUSTNESS: make this depend on camera near/far

	bool consistent_normals = dot(normal, prev_normal) > threshold_normal;
	bool consistent_depth   = abs(depth - prev_depth)  < threshold_depth;

	return consistent_normals && consistent_depth;
}

extern "C" __global__ void kernel_temporal() {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float4 direct   = frame_buffer_direct  [pixel_index];
	float4 indirect = frame_buffer_indirect[pixel_index];

	// First two raw moments of luminance
	float4 moment;
	moment.x = luminance(direct.x,   direct.y,   direct.z);
	moment.y = luminance(indirect.x, indirect.y, indirect.z);
	moment.z = moment.x * moment.x;
	moment.w = moment.y * moment.y;

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	float4 normal_and_depth = tex2D(gbuffer_normal_and_depth, u, v);
	float2 motion           = tex2D(gbuffer_motion,           u, v);

	float3 normal = make_float3(normal_and_depth);
	float  depth  = normal_and_depth.w;

	float u_prev = 0.5f + 0.5f * motion.x;
	float v_prev = 0.5f + 0.5f * motion.y;

	float s_prev = u_prev * float(SCREEN_WIDTH)  - 0.5f;
	float t_prev = v_prev * float(SCREEN_HEIGHT) - 0.5f;
	
	int x_prev = int(s_prev);
	int y_prev = int(t_prev);

	// Calculate bilinear weights
	float fractional_s = s_prev - floor(s_prev);
	float fractional_t = t_prev - floor(t_prev);

	float one_minus_fractional_s = 1.0f - fractional_s;
	float one_minus_fractional_t = 1.0f - fractional_t;

	float w0 = one_minus_fractional_s * one_minus_fractional_t;
	float w1 =           fractional_s * one_minus_fractional_t;
	float w2 = one_minus_fractional_s *           fractional_t;
	float w3 = 1.0f - w0 - w1 - w2;

	float weights[4] = { w0, w1, w2, w3 };

	float consistent_weights[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	float consistent_weights_sum = 0.0f;

	const int2 offsets[4] = {
		{ 0, 0 }, { 1, 0 },
		{ 0, 1 }, { 1, 1 }
	};

	// For each tap in a 2x2 bilinear filter, check if the reprojection is consistent
	// We sum the consistent bilinear weights for normalization purposes later on (weights should always add up to 1)
	for (int tap = 0; tap < 4; tap++) {
		int2 offset = offsets[tap];

		if (is_tap_consistent(x_prev + offset.x, y_prev + offset.y, normal, depth)) {
			float weight = weights[tap];

			consistent_weights[tap] = weight;
			consistent_weights_sum += weight;
		}
	}

	float4 prev_direct   = make_float4(0.0f);
	float4 prev_indirect = make_float4(0.0f);
	float4 prev_moment   = make_float4(0.0f);

	// If we already found at least 1 consistent tap
	if (consistent_weights_sum > 0.0f) {
		// Add consistent taps using their bilinear weight
		for (int tap = 0; tap < 4; tap++) {
			if (consistent_weights[tap] != 0.0f) {
				int2 offset = offsets[tap];
				
				int tap_x = x_prev + offset.x;
				int tap_y = y_prev + offset.y;

				int tap_index = tap_x + tap_y * SCREEN_WIDTH;

				float4 tap_direct   = history_direct  [tap_index];
				float4 tap_indirect = history_indirect[tap_index];
				float4 tap_moment   = history_moment  [tap_index];

				prev_direct   += consistent_weights[tap] * tap_direct;
				prev_indirect += consistent_weights[tap] * tap_indirect;
				prev_moment   += consistent_weights[tap] * tap_moment;
			}
		}

		// Divide by the sum of the consistent weights to renormalize the sum of the consistent weights to 1
		prev_direct   /= consistent_weights_sum;
		prev_indirect /= consistent_weights_sum;
		prev_moment   /= consistent_weights_sum;
	} else {
		// If we haven't yet found a consistent tap in a 2x2 region, try a 3x3 region
        for (int j = -1; j <= 1; j++) {
			for (int i = -1; i <= 1; i++) {
				int tap_x = x_prev + i;
				int tap_y = y_prev + j;

				if (is_tap_consistent(tap_x, tap_y, normal, depth)) {
					int tap_index = tap_x + tap_y * SCREEN_WIDTH;

					prev_direct   += history_direct  [tap_index];
					prev_indirect += history_indirect[tap_index];
					prev_moment   += history_moment  [tap_index];

					consistent_weights_sum += 1.0f;
				}
			}
		}

		if (consistent_weights_sum > 0.0f) {
			prev_direct   /= consistent_weights_sum;
			prev_indirect /= consistent_weights_sum;
			prev_moment   /= consistent_weights_sum;
		}
	}

	if (consistent_weights_sum > 0.0f) {
		int history = ++history_length[pixel_index]; // Increase History Length by 1 step

		float inv_history = 1.0f / float(history);
		float alpha_colour = max(ALPHA_COLOUR, inv_history);
		float alpha_moment = max(ALPHA_COLOUR, inv_history);

		// Integrate using exponential moving average
	 	direct   = alpha_colour * direct   + (1.0f - alpha_colour) * prev_direct;
	 	indirect = alpha_colour * indirect + (1.0f - alpha_colour) * prev_indirect;
		moment   = alpha_moment * moment   + (1.0f - alpha_moment) * prev_moment;
		
		if (history >= 4) {
			float variance_direct   = max(0.0f, moment.z - moment.x * moment.x);
			float variance_indirect = max(0.0f, moment.w - moment.y * moment.y);
			
			// Store the Variance in the alpha channel
			direct.w   = variance_direct;
			indirect.w = variance_indirect;
		}
	} else {
		history_length[pixel_index] = 0; // Reset History Length
	}

	frame_buffer_direct  [pixel_index] = direct;
	frame_buffer_indirect[pixel_index] = indirect;
	frame_buffer_moment  [pixel_index] = moment;
}

extern "C" __global__ void kernel_variance(
	float4 const * colour_direct_in,
	float4 const * colour_indirect_in,
	float4       * colour_direct_out,
	float4       * colour_indirect_out
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	int history = history_length[pixel_index];

	if (history >= 4) {
		// @SPEED
		colour_direct_out  [pixel_index] = colour_direct_in  [pixel_index];
		colour_indirect_out[pixel_index] = colour_indirect_in[pixel_index];

		return;
	}

	// @SPEED: some redundancies here
	float luminance_denom_direct   = 1.0f / (sigma_l + epsilon);
	float luminance_denom_indirect = 1.0f / (sigma_l + epsilon);

	float4 center_colour_direct   = colour_direct_in  [pixel_index];
	float4 center_colour_indirect = colour_indirect_in[pixel_index];

	float center_luminance_direct   = luminance(center_colour_direct.x,   center_colour_direct.y,   center_colour_direct.z);
	float center_luminance_indirect = luminance(center_colour_indirect.x, center_colour_indirect.y, center_colour_indirect.z);

	float4 center_normal_and_depth = tex2D(gbuffer_normal_and_depth, u, v);
	float2 center_depth_gradient   = tex2D(gbuffer_depth_gradient,   u, v);

	float3 center_normal = make_float3(center_normal_and_depth);
	float  center_depth  = center_normal_and_depth.w;

	float sum_weight_direct   = 1.0f;
	float sum_weight_indirect = 1.0f;

	float4 sum_colour_direct   = center_colour_direct;
	float4 sum_colour_indirect = center_colour_indirect;

	float4 sum_moment = make_float4(0.0f);

	const int radius = 3; // 7x7 filter
	
	for (int j = -radius; j <= radius; j++) {
		int tap_y = y + j;

		if (tap_y < 0 || tap_y >= SCREEN_HEIGHT) continue;

		for (int i = -radius; i <= radius; i++) {
			int tap_x = x + i;

			if (tap_x < 0 || tap_x >= SCREEN_WIDTH) continue;

			if (i == 0 && j == 0) continue; // Center pixel is treated separately

			int tap_index = tap_x + tap_y * SCREEN_WIDTH;

			float tap_u = (float(tap_x) + 0.5f) / float(SCREEN_WIDTH);
			float tap_v = (float(tap_y) + 0.5f) / float(SCREEN_HEIGHT);

			float4 colour_direct   = colour_direct_in   [tap_index];
			float4 colour_indirect = colour_indirect_in [tap_index];
			float4 moment          = frame_buffer_moment[tap_index];

			float luminance_direct   = luminance(colour_direct.x,   colour_direct.y,   colour_direct.z);
			float luminance_indirect = luminance(colour_indirect.x, colour_indirect.y, colour_indirect.z);

			float4 normal_and_depth = tex2D(gbuffer_normal_and_depth, tap_u, tap_v);

			float3 normal = make_float3(normal_and_depth);
			float  depth  = normal_and_depth.w;
		
			// @TODO: factor this

			float d = 
				center_depth_gradient.x * float(i) + 
				center_depth_gradient.y * float(j); // ∇z(p)·(p−q)
			//float w_z = (phi_depth == 0.0f) ? 0.0f : abs(center_depth - depth) / phi_depth;
			float w_z = exp(-abs(center_depth - depth) / (sigma_z * abs(d) + epsilon));

			float w_n = pow(max(0.0f, dot(center_normal, normal)), sigma_n);

			float w_l_direct   = exp(-abs(center_luminance_direct   - luminance_direct)   * luminance_denom_direct);
			float w_l_indirect = exp(-abs(center_luminance_indirect - luminance_indirect) * luminance_denom_indirect);

			float w_common   = w_z * w_n;
			float w_direct   = w_common * w_l_direct;
			float w_indirect = w_common * w_l_indirect;

			sum_weight_direct   += w_direct;
			sum_weight_indirect += w_indirect;

			sum_colour_direct   += w_direct   * colour_direct;
			sum_colour_indirect += w_indirect * colour_indirect;

			sum_moment += moment * make_float4(w_direct, w_indirect, w_direct, w_indirect);
		}
	}

	sum_weight_direct   = max(sum_weight_direct,   1e-6f);
	sum_weight_indirect = max(sum_weight_indirect, 1e-6f);
	
	sum_colour_direct   /= sum_weight_direct;
	sum_colour_indirect /= sum_weight_indirect;
	
	sum_moment /= make_float4(sum_weight_direct, sum_weight_indirect, sum_weight_direct, sum_weight_indirect);

	float variance_direct   = max(0.0f, sum_moment.z - sum_moment.x * sum_moment.x);
	float variance_indirect = max(0.0f, sum_moment.w - sum_moment.y * sum_moment.y);

	// float inv_history  = 1.0f / float(history);
	// variance_direct   *= 4.0f * inv_history;
	// variance_indirect *= 4.0f * inv_history;

	sum_colour_direct.w   = variance_direct;
	sum_colour_indirect.w = variance_indirect;
		
	// Store the Variance in the alpha channel
	colour_direct_out  [pixel_index] = sum_colour_direct;
	colour_indirect_out[pixel_index] = sum_colour_indirect;
}

extern "C" __global__ void kernel_atrous(
	float4 const * colour_direct_in,
	float4 const * colour_indirect_in,
	float4       * colour_direct_out,
	float4       * colour_indirect_out,
	int step_size
) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	float variance_blurred_direct   = 0.0f;
	float variance_blurred_indirect = 0.0f;

    const float kernel_gaussian[2][2] = {
        { 1.0f / 4.0f, 1.0f / 8.0f  },
        { 1.0f / 8.0f, 1.0f / 16.0f }
    };

	// Filter Variance using a 3x3 Gaussian Blur
    for (int j = -1; j <= 1; j++) {
		int tap_y = clamp(y + j, 0, SCREEN_HEIGHT - 1);
		
        for (int i = -1; i <= 1; i++) {
            int tap_x = clamp(x + i, 0, SCREEN_WIDTH - 1);

			// Read the Variance of Direct/Indirect Illumination
			// The Variance is stored in the alpha channel (w coordinate)
			float variance_direct   = colour_direct_in  [tap_x + tap_y * SCREEN_WIDTH].w;
			float variance_indirect = colour_indirect_in[tap_x + tap_y * SCREEN_WIDTH].w;

            float kernel_weight = kernel_gaussian[abs(i)][abs(j)];

            variance_blurred_direct   += variance_direct   * kernel_weight;
            variance_blurred_indirect += variance_indirect * kernel_weight;
        }
	}

	// Precompute denominators that are loop invariant
	float luminance_denom_direct   = 1.0f / (sigma_l * sqrt(max(0.0f, variance_blurred_direct))   + epsilon);
	float luminance_denom_indirect = 1.0f / (sigma_l * sqrt(max(0.0f, variance_blurred_indirect)) + epsilon);

	float4 center_colour_direct   = colour_direct_in  [pixel_index];
	float4 center_colour_indirect = colour_indirect_in[pixel_index];

	float center_luminance_direct   = luminance(center_colour_direct.x,   center_colour_direct.y,   center_colour_direct.z);
	float center_luminance_indirect = luminance(center_colour_indirect.x, center_colour_indirect.y, center_colour_indirect.z);

	float4 center_normal_and_depth = tex2D(gbuffer_normal_and_depth, u, v);
	float2 center_depth_gradient   = tex2D(gbuffer_depth_gradient,   u, v);

	float3 center_normal = make_float3(center_normal_and_depth);
	float  center_depth  = center_normal_and_depth.w;

	// float phi_depth = max(max(abs(center_depth_gradient.x), abs(center_depth_gradient.y)), 1e-8) * float(step_size);

	// const float kernel_atrous[3] = {
	// 	3.0f / 8.0f, // 
	// 	1.0f / 4.0f, // 
	// 	1.0f / 16.0f // 
	// };

	const float kernel_atrous[3] = {
		1.0f, 
		2.0f / 3.0f, 
		1.0f / 6.0f 
	};

	float  sum_weight_direct   = 1.0f;
	float  sum_weight_indirect = 1.0f;
	float4 sum_colour_direct   = center_colour_direct;
	float4 sum_colour_indirect = center_colour_indirect;

	// 5x5 À-Trous Filter
	const int radius = 2;

	for (int j = -radius; j <= radius; j++) {
		int tap_y = y + j * step_size;

		if (tap_y < 0 || tap_y >= SCREEN_HEIGHT) continue;

        for (int i = -radius; i <= radius; i++) {
			int tap_x = x + i * step_size;
			
			if (tap_x < 0 || tap_x >= SCREEN_WIDTH) continue;
			
			if (i == 0 && j == 0) continue; // Center pixel is treated separately

			float tap_u = (float(tap_x) + 0.5f) / float(SCREEN_WIDTH);
			float tap_v = (float(tap_y) + 0.5f) / float(SCREEN_HEIGHT);

			float4 colour_direct   = colour_direct_in  [tap_x + tap_y * SCREEN_WIDTH];
			float4 colour_indirect = colour_indirect_in[tap_x + tap_y * SCREEN_WIDTH];

			float luminance_direct   = luminance(colour_direct.x,   colour_direct.y,   colour_direct.z);
			float luminance_indirect = luminance(colour_indirect.x, colour_indirect.y, colour_indirect.z);

			float4 normal_and_depth = tex2D(gbuffer_normal_and_depth, tap_u, tap_v);

			float3 normal = make_float3(normal_and_depth);
			float  depth  = normal_and_depth.w;
			
			float d = 
				center_depth_gradient.x * float(i * step_size) + 
				center_depth_gradient.y * float(j * step_size); // ∇z(p)·(p−q)
			//float w_z = (phi_depth == 0.0f) ? 0.0f : abs(center_depth - depth);
			float w_z = exp(-abs(center_depth - depth) / (sigma_z * abs(d) + epsilon));

			float w_n = powf(max(0.0f, dot(center_normal, normal)), sigma_n);

			float w_l_direct   = exp(-abs(center_luminance_direct   - luminance_direct)   * luminance_denom_direct);
			float w_l_indirect = exp(-abs(center_luminance_indirect - luminance_indirect) * luminance_denom_indirect);

			float kernel_weight = kernel_atrous[abs(i)] * kernel_atrous[abs(j)];

			float w_common   = kernel_weight * w_z * w_n;
			float w_direct   = w_common * w_l_direct;
			float w_indirect = w_common * w_l_indirect;

			sum_weight_direct   += w_direct;
			sum_weight_indirect += w_indirect;

			// Filter Colour using the weights
			// Filter Variance using the square of the weights
			sum_colour_direct   += make_float4(w_direct,   w_direct,   w_direct,   w_direct   * w_direct)   * colour_direct;
			sum_colour_indirect += make_float4(w_indirect, w_indirect, w_indirect, w_indirect * w_indirect) * colour_indirect;
		}
	}

	if (sum_weight_direct > 10e-6f) {
		sum_colour_direct.x /= sum_weight_direct;
		sum_colour_direct.y /= sum_weight_direct;
		sum_colour_direct.z /= sum_weight_direct;
		sum_colour_direct.w /= sum_weight_direct * sum_weight_direct; // Alpha channel contains Variance
	}
	
	if (sum_weight_indirect > 10e-6f) {
		sum_colour_indirect.x /= sum_weight_indirect;
		sum_colour_indirect.y /= sum_weight_indirect;
		sum_colour_indirect.z /= sum_weight_indirect;
		sum_colour_indirect.w /= sum_weight_indirect * sum_weight_indirect; // Alpha channel contains Variance
	}

	colour_direct_out  [pixel_index] = sum_colour_direct;
	colour_indirect_out[pixel_index] = sum_colour_indirect;
	
	const int feedback_iteration = 1;
	
	if (step_size == (1 << feedback_iteration)) {
		history_direct  [pixel_index] = sum_colour_direct;
		history_indirect[pixel_index] = sum_colour_indirect;
	}
}

// Updating the Colour History buffer needs a separate kernel because
// multiple pixels may read from the same texel,
// thus we can only update it after all reads are done
extern "C" __global__ void kernel_finalize(const float4 * colour_direct, const float4 * colour_indirect) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= SCREEN_WIDTH || y >= SCREEN_HEIGHT) return;

	int pixel_index = x + y * SCREEN_WIDTH;

	float4 albedo   = frame_buffer_albedo[pixel_index];
	float4 direct   = colour_direct      [pixel_index];
	float4 indirect = colour_indirect    [pixel_index];

	float4 colour = albedo * (direct + indirect);
	surf2Dwrite(colour, accumulator, x * sizeof(float4), y);

	float4 moment = frame_buffer_moment[pixel_index];

	float u = (float(x) + 0.5f) / float(SCREEN_WIDTH);
	float v = (float(y) + 0.5f) / float(SCREEN_HEIGHT);

	float4 normal_and_depth = tex2D(gbuffer_normal_and_depth, u, v);

#if ATROUS_ITERATIONS == 0
	history_direct  [pixel_index] = direct;
	history_indirect[pixel_index] = indirect;
#endif
	history_moment          [pixel_index] = moment;
	history_normal_and_depth[pixel_index] = normal_and_depth;

	// @SPEED
	// Clear frame buffers for next frame
	frame_buffer_albedo  [pixel_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	frame_buffer_direct  [pixel_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	frame_buffer_indirect[pixel_index] = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
}
