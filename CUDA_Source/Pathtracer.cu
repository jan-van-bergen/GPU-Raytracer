#include <vector_types.h>
#include <corecrt_math.h>

#include "cuda_math.h"

#include "../Common.h"

#include "Tracing.h"
#include "Lighting.h"
#include "Sky.h"
#include "Util.h"

surface<void, 2> frame_buffer;
surface<void, 2> accumulator;

__device__ void frame_buffer_add(int x, int y, const float3 & colour) {
	float4 prev;
	surf2Dread<float4>(&prev, frame_buffer, x * sizeof(float4), y);
	
	surf2Dwrite<float4>(prev + make_float4(colour, 0.0f), frame_buffer, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

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

struct RayBuffer {
	Vector3 origin;
	Vector3 direction;
	
	int   * triangle_id;
	float * u;
	float * v;

	int   * pixel_index;
	Vector3 throughput;

	char * last_material_type;
};

__device__ RayBuffer ray_buffer_extend;
__device__ RayBuffer ray_buffer_shade_diffuse;
__device__ RayBuffer ray_buffer_shade_dielectric;
__device__ RayBuffer ray_buffer_shade_glossy;

struct ShadowRayBuffer {
	Vector3 direction;

	int   * triangle_id;
	float * u;
	float * v;

	int   * pixel_index;
	Vector3 throughput;
};

__device__ ShadowRayBuffer shadow_ray_buffer;

__device__ int N_ext;
__device__ int N_diffuse;
__device__ int N_dielectric;
__device__ int N_glossy;
__device__ int N_shadow;

extern "C" __global__ void kernel_generate(
	int rand_seed,
	float3 camera_position,
	float3 camera_top_left_corner,
	float3 camera_x_axis,
	float3 camera_y_axis
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= SCREEN_WIDTH * SCREEN_HEIGHT) return;

	unsigned seed = (index + rand_seed * 199494991) * 949525949;
	
	const int BLOCK_WIDTH  = 8;
	const int BLOCK_HEIGHT = 4;
	const int BLOCK_SIZE   = BLOCK_WIDTH * BLOCK_HEIGHT;

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

	// Add random value between 0 and 1 so that after averaging we get anti-aliasing
	float u = x + random_float(seed);
	float v = y + random_float(seed);

	ASSERT(pixel_index < SCREEN_WIDTH * SCREEN_HEIGHT, "Pixel should be on screen");

	// Create primary Ray that starts at the Camera's position and goes through the current pixel
	ray_buffer_extend.origin.from_float3(index, camera_position);
	ray_buffer_extend.direction.from_float3(index, normalize(camera_top_left_corner
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
	if (index >= N_ext) return;

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
	bvh_trace(ray, hit);

	int ray_pixel_index = ray_buffer_extend.pixel_index[index];

	if (hit.t == INFINITY) {
		int x = ray_pixel_index % SCREEN_WIDTH;
		int y = ray_pixel_index / SCREEN_WIDTH; 

		frame_buffer_add(x, y, ray_buffer_extend.throughput.to_float3(index) * sample_sky(ray_direction));

		return;
	}

	float3 ray_throughput = ray_buffer_extend.throughput.to_float3(index);

	unsigned seed = (ray_pixel_index + rand_seed * 906313609) * 341828143;

	// Russian Roulette termination
	if (bounce > 3) {
		float p_survive = clamp(fmaxf(ray_throughput.x, fmaxf(ray_throughput.y, ray_throughput.z)), 0.0f, 1.0f);
		if (random_float(seed) > p_survive) {
			return;
		}

		ray_throughput /= p_survive;
	}

	// Get the Material of the Triangle we hit
	const Material & material = materials[triangles_material_id[hit.triangle_id]];

	if (material.type == Material::Type::LIGHT) {
		if (ray_buffer_extend.last_material_type[index] == Material::Type::DIELECTRIC ||
			ray_buffer_extend.last_material_type[index] == Material::Type::GLOSSY) {
			int x = ray_pixel_index % SCREEN_WIDTH;
			int y = ray_pixel_index / SCREEN_WIDTH; 

			frame_buffer_add(x, y, ray_throughput * material.emittance);
		}
	} else if (material.type == Material::Type::DIFFUSE) {
		int index_out = atomic_agg_inc(&N_diffuse);

		// ray_buffer_shade_diffuse.direction[index_out] = ray_direction;

		assert(!isnan(hit.u) && !isnan(hit.v));

		ray_buffer_shade_diffuse.triangle_id[index_out] = hit.triangle_id;
		ray_buffer_shade_diffuse.u[index_out] = hit.u;
		ray_buffer_shade_diffuse.v[index_out] = hit.v;

		ray_buffer_shade_diffuse.pixel_index[index_out] = ray_buffer_extend.pixel_index[index];
		ray_buffer_shade_diffuse.throughput.from_float3(index_out, ray_throughput);
	} else if (material.type == Material::Type::DIELECTRIC) {
		int index_out = atomic_agg_inc(&N_dielectric);

		ray_buffer_shade_dielectric.direction.from_float3(index_out, ray_direction);

		ray_buffer_shade_dielectric.triangle_id[index_out] = hit.triangle_id;
		ray_buffer_shade_dielectric.u[index_out] = hit.u;
		ray_buffer_shade_dielectric.v[index_out] = hit.v;

		ray_buffer_shade_dielectric.pixel_index[index_out] = ray_buffer_extend.pixel_index[index];
		ray_buffer_shade_dielectric.throughput.from_float3(index_out, ray_throughput);
	} else if (material.type == Material::Type::GLOSSY) {
		int index_out = atomic_agg_inc(&N_glossy);

		ray_buffer_shade_glossy.direction.from_float3(index_out, ray_direction);

		ray_buffer_shade_glossy.triangle_id[index_out] = hit.triangle_id;
		ray_buffer_shade_glossy.u[index_out] = hit.u;
		ray_buffer_shade_glossy.v[index_out] = hit.v;

		ray_buffer_shade_glossy.pixel_index[index_out] = ray_buffer_extend.pixel_index[index];
		ray_buffer_shade_glossy.throughput.from_float3(index_out, ray_throughput);
	}
}

extern "C" __global__ void kernel_shade_diffuse(int rand_seed) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N_diffuse) return;

	// float3 ray_direction = ray_buffer_shade_diffuse.direction[index];

	int   ray_triangle_id = ray_buffer_shade_diffuse.triangle_id[index];
	float ray_u = ray_buffer_shade_diffuse.u[index];
	float ray_v = ray_buffer_shade_diffuse.v[index];

	int    ray_pixel_index = ray_buffer_shade_diffuse.pixel_index[index];
	float3 ray_throughput  = ray_buffer_shade_diffuse.throughput.to_float3(index);

	ASSERT(ray_triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = (ray_pixel_index + rand_seed * 312080213) * 781939187;

	const Material & material = materials[triangles_material_id[ray_triangle_id]];

	ASSERT(material.type == Material::Type::DIFFUSE, "Material should be diffuse in this Kernel");

	if (light_count > 0) {
		int shadow_ray_index = atomic_agg_inc(&N_shadow);

		// shadow_ray_buffer.direction not required for diffuse BRDF evaluation

		shadow_ray_buffer.triangle_id[shadow_ray_index] = ray_triangle_id;
		shadow_ray_buffer.u[shadow_ray_index] = ray_u;
		shadow_ray_buffer.v[shadow_ray_index] = ray_v;

		shadow_ray_buffer.pixel_index[shadow_ray_index]  = ray_pixel_index;
		shadow_ray_buffer.throughput.from_float3(shadow_ray_index, ray_throughput);
	}

	float3 hit_point     = barycentric(ray_u, ray_v, triangles_position0 [ray_triangle_id], triangles_position_edge1 [ray_triangle_id], triangles_position_edge2 [ray_triangle_id]);
	float3 hit_normal    = barycentric(ray_u, ray_v, triangles_normal0   [ray_triangle_id], triangles_normal_edge1   [ray_triangle_id], triangles_normal_edge2   [ray_triangle_id]);
	float2 hit_tex_coord = barycentric(ray_u, ray_v, triangles_tex_coord0[ray_triangle_id], triangles_tex_coord_edge1[ray_triangle_id], triangles_tex_coord_edge2[ray_triangle_id]);

	// if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	int index_out = atomic_agg_inc(&N_ext);

	float3 direction = cosine_weighted_diffuse_reflection(seed, hit_normal);

	ray_buffer_extend.origin.from_float3(index_out, hit_point);
	ray_buffer_extend.direction.from_float3(index_out, direction);

	float3 throughput = ray_throughput * material.albedo(hit_tex_coord.x, hit_tex_coord.y);

	ray_buffer_extend.pixel_index[index_out]  = ray_pixel_index;
	ray_buffer_extend.throughput.from_float3(index_out, throughput);

	ray_buffer_extend.last_material_type[index_out] = char(Material::Type::DIFFUSE);
}

extern "C" __global__ void kernel_shade_dielectric(int rand_seed) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N_dielectric) return;

	float3 ray_direction = ray_buffer_shade_dielectric.direction.to_float3(index);

	int   ray_triangle_id = ray_buffer_shade_dielectric.triangle_id[index];
	float ray_u = ray_buffer_shade_dielectric.u[index];
	float ray_v = ray_buffer_shade_dielectric.v[index];

	int    ray_pixel_index = ray_buffer_shade_dielectric.pixel_index[index];
	float3 ray_throughput  = ray_buffer_shade_dielectric.throughput.to_float3(index);

	ASSERT(ray_triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = (ray_pixel_index + rand_seed * 312080213) * 781939187;

	const Material & material = materials[triangles_material_id[ray_triangle_id]];

	ASSERT(material.type == Material::Type::DIELECTRIC, "Material should be dielectric in this Kernel");

	float3 hit_point     = barycentric(ray_u, ray_v, triangles_position0 [ray_triangle_id], triangles_position_edge1 [ray_triangle_id], triangles_position_edge2 [ray_triangle_id]);
	float3 hit_normal    = barycentric(ray_u, ray_v, triangles_normal0   [ray_triangle_id], triangles_normal_edge1   [ray_triangle_id], triangles_normal_edge2   [ray_triangle_id]);
	float2 hit_tex_coord = barycentric(ray_u, ray_v, triangles_tex_coord0[ray_triangle_id], triangles_tex_coord_edge1[ray_triangle_id], triangles_tex_coord_edge2[ray_triangle_id]);

	// if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	int index_out = atomic_agg_inc(&N_ext);

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

		if (random_float(seed) < F_r) {
			direction = direction_reflected;
		} else {
			direction = direction_refracted;
		}
	}

	ray_buffer_extend.origin.from_float3(index_out, hit_point);
	ray_buffer_extend.direction.from_float3(index_out, direction);

	ray_buffer_extend.pixel_index[index_out]  = ray_pixel_index;
	ray_buffer_extend.throughput.from_float3(index_out, ray_throughput);

	ray_buffer_extend.last_material_type[index_out] = char(Material::Type::DIELECTRIC);
}

extern "C" __global__ void kernel_shade_glossy(int rand_seed) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N_glossy) return;

	float3 direction_in = -ray_buffer_shade_glossy.direction.to_float3(index);

	int   ray_triangle_id = ray_buffer_shade_glossy.triangle_id[index];
	float ray_u = ray_buffer_shade_glossy.u[index];
	float ray_v = ray_buffer_shade_glossy.v[index];

	int    ray_pixel_index = ray_buffer_shade_glossy.pixel_index[index];
	float3 ray_throughput  = ray_buffer_shade_glossy.throughput.to_float3(index);

	ASSERT(ray_triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = (ray_pixel_index + rand_seed * 312080213) * 781939187;

	const Material & material = materials[triangles_material_id[ray_triangle_id]];

	ASSERT(material.type == Material::Type::GLOSSY, "Material should be glossy in this Kernel");

	if (light_count > 0) {
		// int shadow_ray_index = atomic_agg_inc(&N_shadow);

		// shadow_ray_buffer.triangle_id[shadow_ray_index] = ray_triangle_id;
		// shadow_ray_buffer.u[shadow_ray_index] = ray_u;
		// shadow_ray_buffer.v[shadow_ray_index] = ray_v;

		// shadow_ray_buffer.pixel_index[shadow_ray_index] = ray_pixel_index;
		// shadow_ray_buffer.throughput[shadow_ray_index]  = ray_throughput;
	}

	float3 hit_point     = barycentric(ray_u, ray_v, triangles_position0 [ray_triangle_id], triangles_position_edge1 [ray_triangle_id], triangles_position_edge2 [ray_triangle_id]);
	float3 hit_normal    = barycentric(ray_u, ray_v, triangles_normal0   [ray_triangle_id], triangles_normal_edge1   [ray_triangle_id], triangles_normal_edge2   [ray_triangle_id]);
	float2 hit_tex_coord = barycentric(ray_u, ray_v, triangles_tex_coord0[ray_triangle_id], triangles_tex_coord_edge1[ray_triangle_id], triangles_tex_coord_edge2[ray_triangle_id]);

	if (dot(direction_in, hit_normal) < 0.0f) hit_normal = -hit_normal;

	// Slightly widen the distribution to prevent the weights from becoming too large (see Walter et al. 2007)
	float alpha = (1.2f - 0.2f * sqrt(dot(direction_in, hit_normal))) * material.roughness;
	
	// Sample normal distribution in spherical coordinates
	float theta = atan(sqrt(-alpha * alpha * log(random_float(seed) + 1e-8f)));
	float phi   = TWO_PI * random_float(seed);

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

	float g = 
		beckmann_g1(direction_in,  micro_normal_world, hit_normal, alpha) * 
		beckmann_g1(direction_out, micro_normal_world, hit_normal, alpha);
	float weight = abs(dot(direction_in, micro_normal_world)) * g / abs(dot(direction_in, hit_normal) * dot(micro_normal_world, hit_normal));

	int index_out = atomic_agg_inc(&N_ext);

	ray_buffer_extend.origin.from_float3(index_out, hit_point);
	ray_buffer_extend.direction.from_float3(index_out, direction_out);

	ray_buffer_extend.pixel_index[index_out]  = ray_pixel_index;
	ray_buffer_extend.throughput.from_float3(index_out, ray_throughput * material.albedo(hit_tex_coord.x, hit_tex_coord.y) * weight);

	ray_buffer_extend.last_material_type[index_out] = char(Material::Type::GLOSSY);
}

extern "C" __global__ void kernel_connect(int rand_seed) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= N_shadow) return;

	int   ray_triangle_id = shadow_ray_buffer.triangle_id[index];
	float ray_u = shadow_ray_buffer.u[index];
	float ray_v = shadow_ray_buffer.v[index];

	int    ray_pixel_index = shadow_ray_buffer.pixel_index[index];
	float3 ray_throughput  = shadow_ray_buffer.throughput.to_float3(index);

	unsigned seed = (ray_pixel_index + rand_seed * 390292093) * 162898261;

	// Pick a random light emitting triangle
	int light_triangle_id = light_indices[rand_xorshift(seed) % light_count];

	ASSERT(length(materials[triangles_material_id[light_triangle_id]].emittance) > 0.0f, "Material was not emissive!\n");

	// Pick a random point on the triangle using random barycentric coordinates
	float u = random_float(seed);
	float v = random_float(seed);

	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}

	float3 random_point_on_light = barycentric(u,     v,     triangles_position0[light_triangle_id], triangles_position_edge1[light_triangle_id], triangles_position_edge2[light_triangle_id]);
	float3 hit_point             = barycentric(ray_u, ray_v, triangles_position0[ray_triangle_id],   triangles_position_edge1[ray_triangle_id],   triangles_position_edge2[ray_triangle_id]);
	float3 hit_normal            = barycentric(ray_u, ray_v, triangles_normal0  [ray_triangle_id],   triangles_normal_edge1  [ray_triangle_id],   triangles_normal_edge2  [ray_triangle_id]);

	float3 to_light = random_point_on_light - hit_point;
	float distance_to_light_squared = dot(to_light, to_light);
	float distance_to_light         = sqrtf(distance_to_light_squared);

	// Normalize the vector to the light
	to_light /= distance_to_light;

	float3 light_normal = barycentric(u, v, triangles_normal0[light_triangle_id], triangles_normal_edge1[light_triangle_id], triangles_normal_edge2[light_triangle_id]);

	float cos_o = -dot(to_light, light_normal);
	float cos_i =  dot(to_light, hit_normal);

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
		if (!bvh_intersect(shadow_ray, distance_to_light - EPSILON)) {
			const Material & hit_material   = materials[triangles_material_id[ray_triangle_id]];
			const Material & light_material = materials[triangles_material_id[light_triangle_id]];

			float2 hit_tex_coord = barycentric(ray_u, ray_v, triangles_tex_coord0[ray_triangle_id], triangles_tex_coord_edge1[ray_triangle_id], triangles_tex_coord_edge2[ray_triangle_id]);

			float3 brdf = hit_material.albedo(hit_tex_coord.x, hit_tex_coord.y) * ONE_OVER_PI;

			float light_area = 0.5f * length(cross(
				triangles_position_edge1[light_triangle_id], 
				triangles_position_edge2[light_triangle_id]
			));
			float solid_angle = (cos_o * light_area) / distance_to_light_squared;

			int x = ray_pixel_index % SCREEN_WIDTH;
			int y = ray_pixel_index / SCREEN_WIDTH; 
		
			frame_buffer_add(x, y, ray_throughput * brdf * light_count * light_material.emittance * solid_angle * cos_i);
		}
	}
}

extern "C" __global__ void kernel_accumulate(float frames_since_camera_moved) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	float4 colour;
	surf2Dread<float4>(&colour, frame_buffer, x * sizeof(float4), y);
	
	float4 colour_out;
	if (frames_since_camera_moved > 0.0f) {
		float4 prev;
		surf2Dread<float4>(&prev, accumulator, x * sizeof(float4), y);

		// Take average over n samples by weighing the current content of the framebuffer by (n-1) and the new sample by 1
		colour_out = (prev * (frames_since_camera_moved - 1.0f) + colour) / frames_since_camera_moved;
	} else {
		colour_out = colour;
	}

	surf2Dwrite<float4>(colour_out, accumulator, x * sizeof(float4), y, cudaBoundaryModeClamp);

	// Clear frame buffer for next frame
	surf2Dwrite<float4>(make_float4(0.0f, 0.0f, 0.0f, 1.0f), frame_buffer, x * sizeof(float4), y, cudaBoundaryModeClamp);
}
