#include "cudart/vector_types.h"
#include "cudart/cuda_math.h"

#include "Common.h"

__device__ __constant__ int screen_width;
__device__ __constant__ int screen_pitch;
__device__ __constant__ int screen_height;

__device__ __constant__ Settings settings;

#include "Util.h"
#include "Shading.h"
#include "Sky.h"
#include "Random.h"

#define INFINITY ((float)(1e+300 * 1e+300))

// Frame Buffers
__device__ float4 * frame_buffer_albedo;
__device__ float4 * frame_buffer_direct;
__device__ float4 * frame_buffer_indirect;

__device__ float4 * frame_buffer_moment;

// GBuffers (OpenGL resource-mapped textures)
__device__ Texture<float4> gbuffer_normal_and_depth;
__device__ Texture<float2> gbuffer_uv;
__device__ Texture<float4> gbuffer_uv_gradient;
__device__ Texture<int2>   gbuffer_mesh_id_and_triangle_id;
__device__ Texture<float2> gbuffer_screen_position_prev;
__device__ Texture<float2> gbuffer_depth_gradient;

// SVGF History Buffers (Temporally Integrated)
__device__ int    * history_length;
__device__ float4 * history_direct;
__device__ float4 * history_indirect;
__device__ float4 * history_moment;
__device__ float4 * history_normal_and_depth;

// Used for Temporal Anti-Aliasing
__device__ float4 * taa_frame_curr;
__device__ float4 * taa_frame_prev;

// Final Frame buffer, shared with OpenGL
__device__ Surface<float4> accumulator; 

#include "SVGF.h"
#include "TAA.h"

// Vector3 buffer in SoA layout
struct Vector3_SoA {
	float * x;
	float * y;
	float * z;

	__device__ void set(int index, const float3 & vector) {
		x[index] = vector.x;
		y[index] = vector.y;
		z[index] = vector.z;
	}

	__device__ float3 get(int index) const {
		return make_float3(
			x[index],
			y[index],
			z[index]
		);
	}
};

struct RayHit {
	float t;
	float u, v;

	int mesh_id;
	int triangle_id;
};

struct HitBuffer {
	uint4 * hits;

	__device__ void set(int index, const RayHit & ray_hit) {
		unsigned uv = int(ray_hit.u * 65535.0f) | (int(ray_hit.v * 65535.0f) << 16);

		hits[index] = make_uint4(ray_hit.mesh_id, ray_hit.triangle_id, float_as_uint(ray_hit.t), uv);
	}

	__device__ RayHit get(int index) const {
		uint4 hit = __ldg(&hits[index]);

		RayHit ray_hit;

		ray_hit.mesh_id     = hit.x;
		ray_hit.triangle_id = hit.y;
		
		ray_hit.t = uint_as_float(hit.z);

		ray_hit.u = float(hit.w & 0xffff) / 65535.0f;
		ray_hit.v = float(hit.w >> 16)    / 65535.0f;

		return ray_hit;
	}
};

// Input to the Trace and Sort Kernels in SoA layout
struct TraceBuffer {
	Vector3_SoA origin;
	Vector3_SoA direction;

#if ENABLE_MIPMAPPING
	float2 * cone;
#endif

	HitBuffer hits;

	int       * pixel_index_and_last_material; // Last material in 2 highest bits, pixel index in lowest 30
	Vector3_SoA throughput;

	float * last_pdf;
};

// Input to the various Shade Kernels in SoA layout
struct MaterialBuffer {
	Vector3_SoA direction;	

#if ENABLE_MIPMAPPING
	float2 * cone;
#endif

	HitBuffer hits;

	int       * pixel_index;
	Vector3_SoA throughput;
};

// Input to the Shadow Trace Kernel in SoA layout
struct ShadowRayBuffer {
	Vector3_SoA ray_origin;
	Vector3_SoA ray_direction;

	float * max_distance;

	int       * pixel_index;
	Vector3_SoA illumination;
};

__device__ TraceBuffer     ray_buffer_trace;
__device__ MaterialBuffer  ray_buffer_shade_diffuse;
__device__ MaterialBuffer  ray_buffer_shade_dielectric;
__device__ MaterialBuffer  ray_buffer_shade_glossy;
__device__ ShadowRayBuffer ray_buffer_shadow;

// Number of elements in each Buffer
// Sizes are stored for ALL bounces so we only have to reset these
// values back to 0 after every frame, instead of after every bounce
struct BufferSizes {
	int trace     [MAX_BOUNCES];
	int diffuse   [MAX_BOUNCES];
	int dielectric[MAX_BOUNCES];
	int glossy    [MAX_BOUNCES];
	int shadow    [MAX_BOUNCES];

	// Global counters for tracing kernels
	int rays_retired       [MAX_BOUNCES];
	int rays_retired_shadow[MAX_BOUNCES];
} __device__ buffer_sizes;

struct Camera {
	float3 position;
	float3 bottom_left_corner;
	float3 x_axis;
	float3 y_axis;
	float  pixel_spread_angle;
} __device__ __constant__ camera;

#include "Tracing.h"
#include "Mipmap.h"

// Sends the rasterized GBuffer to the right Material kernels,
// as if the primary Rays they were Raytraced 
extern "C" __global__ void kernel_primary(
	int rand_seed,
	int sample_index,
	int pixel_offset,
	int pixel_count,
	bool jitter
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= pixel_count) return;

	int index_offset = index + pixel_offset;
	int x = index_offset % screen_width;
	int y = index_offset / screen_width;

	int pixel_index = x + y * screen_pitch;

	unsigned seed = wang_hash(pixel_index ^ rand_seed);

	float u_screenspace = float(x) + 0.5f;
	float v_screenspace = float(y) + 0.5f;

	float u = u_screenspace / float(screen_width);
	float v = v_screenspace / float(screen_height);

	float2 uv          = gbuffer_uv         .get(u, v);
	float4 uv_gradient = gbuffer_uv_gradient.get(u, v);

	int2 mesh_id_and_triangle_id = gbuffer_mesh_id_and_triangle_id.get(u, v);
	int mesh_id     = mesh_id_and_triangle_id.x;
	int triangle_id = mesh_id_and_triangle_id.y - 1;

	float dx = 0.0f;
	float dy = 0.0f;
	
	if (jitter) {
		// Jitter the barycentric coordinates in screen space using their screen space differentials
		dx = random_float_heitz(x, y, sample_index, 0, 0, seed) - 0.5f;
		dy = random_float_heitz(x, y, sample_index, 0, 1, seed) - 0.5f;

		uv.x = saturate(uv.x + uv_gradient.x * dx + uv_gradient.z * dy);
		uv.y = saturate(uv.y + uv_gradient.y * dx + uv_gradient.w * dy);
	}

	float2 pixel_coord = make_float2(u_screenspace + dx, v_screenspace + dy);

	float3 ray_direction = normalize(camera.bottom_left_corner + pixel_coord.x * camera.x_axis + pixel_coord.y * camera.y_axis);

	// Triangle ID -1 means no hit
	if (triangle_id == -1) {
		if (settings.demodulate_albedo || settings.enable_svgf) {
			frame_buffer_albedo[pixel_index] = make_float4(1.0f);
		}
		frame_buffer_direct[pixel_index] = make_float4(sample_sky(ray_direction));

		return;
	}

	RayHit ray_hit;
	ray_hit.mesh_id     = mesh_id;
	ray_hit.triangle_id = triangle_id;
	ray_hit.t = 0.0f;
	ray_hit.u = uv.x;
	ray_hit.v = uv.y;

	const Material & material = materials[triangle_get_material_id(triangle_id)];

	// Decide which Kernel to invoke, based on Material Type
	switch (material.type) {
		case Material::Type::LIGHT: {
			// Terminate Path
			if (settings.demodulate_albedo || settings.enable_svgf) {
				frame_buffer_albedo[pixel_index] = make_float4(1.0f);
			}
			frame_buffer_direct[pixel_index] = make_float4(material.emission);

			break;
		}
		
		case Material::Type::DIFFUSE: {
			int index_out = atomic_agg_inc(&buffer_sizes.diffuse[0]);

			ray_buffer_shade_diffuse.direction.set(index_out, ray_direction);

			ray_buffer_shade_diffuse.hits.set(index_out, ray_hit);

			ray_buffer_shade_diffuse.pixel_index[index_out] = pixel_index;
			ray_buffer_shade_diffuse.throughput.set(index_out, make_float3(1.0f));

			break;
		}

		case Material::Type::DIELECTRIC: {
			int index_out = atomic_agg_inc(&buffer_sizes.dielectric[0]);

			ray_buffer_shade_dielectric.direction.set(index_out, ray_direction);

			ray_buffer_shade_dielectric.hits.set(index_out, ray_hit);

			ray_buffer_shade_dielectric.pixel_index[index_out] = pixel_index;
			ray_buffer_shade_dielectric.throughput.set(index_out, make_float3(1.0f));
			
			break;
		}

		case Material::Type::GLOSSY: {
			int index_out = atomic_agg_inc(&buffer_sizes.glossy[0]);

			ray_buffer_shade_glossy.direction.set(index_out, ray_direction);

			ray_buffer_shade_glossy.hits.set(index_out, ray_hit);

			ray_buffer_shade_glossy.pixel_index[index_out] = pixel_index;
			ray_buffer_shade_glossy.throughput.set(index_out, make_float3(1.0f));
			
			break;
		}
	}
}

extern "C" __global__ void kernel_generate(
	int rand_seed,
	int sample_index,
	int pixel_offset,
	int pixel_count
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= pixel_count) return;

	int index_offset = index + pixel_offset;
	int x = index_offset % screen_width;
	int y = index_offset / screen_width;

	unsigned seed = wang_hash(index_offset ^ rand_seed);

	int pixel_index = x + y * screen_pitch;
	ASSERT(pixel_index < screen_pitch * screen_height, "Pixel should fit inside the buffer");

	float u0 = random_float_xorshift(seed);
	float u1 = random_float_xorshift(seed);
	float u2 = random_float_heitz(x, y, sample_index, 0, 0, seed);
	float u3 = random_float_heitz(x, y, sample_index, 0, 1, seed);

	float2 jitter;

	switch (settings.reconstruction_filter) {
		case ReconstructionFilter::BOX: {
			jitter.x = u1;
			jitter.y = u2;

			break;
		}

		case ReconstructionFilter::GAUSSIAN: {
			float2 gaussians = box_muller(u1, u2);

			jitter.x = 0.5f + 0.5f * gaussians.x;
			jitter.y = 0.5f + 0.5f * gaussians.y;
		
			break;
		}
	}

	float x_jittered = float(x) + jitter.x;
	float y_jittered = float(y) + jitter.y;

	float3 focal_point = settings.camera_focal_distance * normalize(camera.bottom_left_corner + x_jittered * camera.x_axis + y_jittered * camera.y_axis);
	float2 lens_point = 0.5f * settings.camera_aperture * random_point_in_disk(u2, u3);

	float3 offset = camera.x_axis * lens_point.x + camera.y_axis * lens_point.y;
	float3 direction = normalize(focal_point - offset);

	// Create primary Ray that starts at the Camera's position and goes through the current pixel
	ray_buffer_trace.origin   .set(index, camera.position + offset);
	ray_buffer_trace.direction.set(index, direction);

	ray_buffer_trace.pixel_index_and_last_material[index] = pixel_index | int(Material::Type::DIELECTRIC) << 30;
	ray_buffer_trace.throughput.set(index, make_float3(1.0f));
}

extern "C" __global__ void kernel_trace(int bounce) {
	bvh_trace(buffer_sizes.trace[bounce], &buffer_sizes.rays_retired[bounce]);
}

extern "C" __global__ void kernel_sort(int rand_seed, int bounce) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.trace[bounce]) return;

	float3 ray_origin    = ray_buffer_trace.origin   .get(index);
	float3 ray_direction = ray_buffer_trace.direction.get(index);

	RayHit hit = ray_buffer_trace.hits.get(index);

	unsigned ray_pixel_index_and_last_material = ray_buffer_trace.pixel_index_and_last_material[index];
	int      ray_pixel_index = ray_pixel_index_and_last_material & ~(0b11 << 30);

	Material::Type last_material_type = Material::Type(ray_pixel_index_and_last_material >> 30);

	float3 ray_throughput = ray_buffer_trace.throughput.get(index);
	
	// If we didn't hit anything, sample the Sky
	if (hit.triangle_id == -1) {
		float3 illumination = ray_throughput * sample_sky(ray_direction);

		if (bounce == 0) {
			if (settings.demodulate_albedo || settings.enable_svgf) {
				frame_buffer_albedo[ray_pixel_index] = make_float4(1.0f);
			}
			frame_buffer_direct[ray_pixel_index] = make_float4(illumination);
		} else if (bounce == 1) {
			frame_buffer_direct[ray_pixel_index] += make_float4(illumination);
		} else {
			frame_buffer_indirect[ray_pixel_index] += make_float4(illumination);
		}

		return;
	}

	// Get the Material of the Triangle we hit
	const Material & material = materials[triangle_get_material_id(hit.triangle_id)];

	if (material.type == Material::Type::LIGHT) {
		bool no_mis = true;
		if (settings.enable_next_event_estimation) {
			no_mis = 
				(last_material_type == Material::Type::DIELECTRIC) ||
				(last_material_type == Material::Type::GLOSSY && material.roughness < ROUGHNESS_CUTOFF);
		}

		if (no_mis) {
			float3 illumination = ray_throughput * material.emission;

			if (bounce == 0) {
				if (settings.demodulate_albedo || settings.enable_svgf) {
					frame_buffer_albedo[ray_pixel_index] = make_float4(1.0f);
				}
				frame_buffer_direct[ray_pixel_index] = make_float4(material.emission);
			} else if (bounce == 1) {
				frame_buffer_direct[ray_pixel_index] += make_float4(illumination);
			} else {
				frame_buffer_indirect[ray_pixel_index] += make_float4(illumination);
			}

			return;
		}

		if (settings.enable_multiple_importance_sampling) {
			// Obtain the Light's position and normal
			TrianglePosNor light = triangle_get_positions_and_normals(hit.triangle_id);

			float3 light_point;
			float3 light_normal;
			triangle_barycentric(light, hit.u, hit.v, light_point, light_normal);

			// Transform into world space
			Matrix3x4 world = mesh_get_transform(hit.mesh_id);
			matrix3x4_transform_position (world, light_point);
			matrix3x4_transform_direction(world, light_normal);

			float3 to_light = light_point - ray_origin;;
			float distance_to_light_squared = dot(to_light, to_light);
			float distance_to_light         = sqrtf(distance_to_light_squared);

			to_light /= distance_to_light; // Normalize

			float cos_o = fabsf(dot(to_light, light_normal));
			// if (cos_o <= 0.0f) return;

			float light_area = 0.5f * length(cross(light.position_edge_1, light.position_edge_2));
			
			float brdf_pdf = ray_buffer_trace.last_pdf[index];

			float light_select_pdf = light_area / light_total_area;
			float light_pdf = light_select_pdf * distance_to_light_squared / (cos_o * light_area); // Convert solid angle measure

			float mis_pdf = brdf_pdf + light_pdf;

			float3 illumination = ray_throughput * material.emission * brdf_pdf / mis_pdf;

			if (bounce == 1) {
				frame_buffer_direct[ray_pixel_index] += make_float4(illumination);
			} else {
				frame_buffer_indirect[ray_pixel_index] += make_float4(illumination);
			}
		}

		return;
	}

	unsigned seed = wang_hash(index ^ rand_seed);

	// Russian Roulette
	float p_survive = saturate(fmaxf(ray_throughput.x, fmaxf(ray_throughput.y, ray_throughput.z)));
	if (random_float_xorshift(seed) > p_survive) {
		return;
	}

	ray_throughput /= p_survive;

	switch (material.type) {
		case Material::Type::DIFFUSE: {
			int index_out = atomic_agg_inc(&buffer_sizes.diffuse[bounce]);

			ray_buffer_shade_diffuse.direction.set(index_out, ray_direction);

#if ENABLE_MIPMAPPING
			if (bounce > 0) ray_buffer_shade_diffuse.cone[index_out] = ray_buffer_trace.cone[index];
#endif
			ray_buffer_shade_diffuse.hits.set(index_out, hit);
			
			ray_buffer_shade_diffuse.pixel_index[index_out] = ray_pixel_index;
			ray_buffer_shade_diffuse.throughput.set(index_out, ray_throughput);

			break;
		}

		case Material::Type::DIELECTRIC: {
			int index_out = atomic_agg_inc(&buffer_sizes.dielectric[bounce]);

			ray_buffer_shade_dielectric.direction.set(index_out, ray_direction);

#if ENABLE_MIPMAPPING
			if (bounce > 0) ray_buffer_shade_dielectric.cone[index_out] = ray_buffer_trace.cone[index];
#endif
			ray_buffer_shade_dielectric.hits.set(index_out, hit);

			ray_buffer_shade_dielectric.pixel_index[index_out] = ray_pixel_index;
			ray_buffer_shade_dielectric.throughput.set(index_out, ray_throughput);

			break;
		}

		case Material::Type::GLOSSY: {
			int index_out = atomic_agg_inc(&buffer_sizes.glossy[bounce]);

			ray_buffer_shade_glossy.direction.set(index_out, ray_direction);

#if ENABLE_MIPMAPPING
			if (bounce > 0) ray_buffer_shade_glossy.cone[index_out] = ray_buffer_trace.cone[index];
#endif
			ray_buffer_shade_glossy.hits.set(index_out, hit);

			ray_buffer_shade_glossy.pixel_index[index_out] = ray_pixel_index;
			ray_buffer_shade_glossy.throughput.set(index_out, ray_throughput);

			break;
		}
	}
}

extern "C" __global__ void kernel_shade_diffuse(int rand_seed, int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.diffuse[bounce]) return;

	float3 ray_direction = ray_buffer_shade_diffuse.direction.get(index);
	RayHit hit           = ray_buffer_shade_diffuse.hits     .get(index);

	int ray_pixel_index = ray_buffer_shade_diffuse.pixel_index[index];
	int x = ray_pixel_index % screen_pitch;
	int y = ray_pixel_index / screen_pitch;

	float3 ray_throughput = ray_buffer_shade_diffuse.throughput.get(index);

	ASSERT(hit.triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = wang_hash(index ^ rand_seed);

	const Material & material = materials[triangle_get_material_id(hit.triangle_id)];

	ASSERT(material.type == Material::Type::DIFFUSE, "Material should be diffuse in this Kernel");

	// Obtain hit Triangle position, normal, and texture coordinates
	TrianglePosNorTex hit_triangle = triangle_get_positions_normals_and_tex_coords(hit.triangle_id);

	float3 hit_point;
	float3 hit_normal;
	float2 hit_tex_coord;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, hit_normal, hit_tex_coord);

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, hit_normal);

	if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	float3 albedo;

#if ENABLE_MIPMAPPING
	float cone_angle, cone_width;

	float3 geometric_normal = normalize(cross(hit_triangle.position_edge_1, hit_triangle.position_edge_2));

	if (bounce == 0) {
		cone_angle = camera.pixel_spread_angle;
		cone_width = cone_angle * hit.t;

		float3 ellipse_axis_1, ellipse_axis_2; ray_cone_get_ellipse_axes(ray_direction, geometric_normal, cone_width, ellipse_axis_1, ellipse_axis_2);

		float2 gradient_1, gradient_2; ray_cone_get_texture_gradients(
			geometric_normal,
			hit_triangle.position_0,  hit_triangle.position_edge_1,  hit_triangle.position_edge_2,
			hit_triangle.tex_coord_0, hit_triangle.tex_coord_edge_1, hit_triangle.tex_coord_edge_2,
			hit_point, hit_tex_coord,
			ellipse_axis_1, ellipse_axis_2,
			gradient_1, gradient_2
		);

		// Anisotropic sampling
		albedo = material.albedo(hit_tex_coord.x, hit_tex_coord.y, gradient_1, gradient_2);
	} else {
		float2 cone = ray_buffer_shade_diffuse.cone[index];
		cone_angle = cone.x;
		cone_width = cone.y + cone_angle * hit.t;

		// Trilinear sampling
		float lod = triangle_get_lod(hit.triangle_id) + ray_cone_get_lod(ray_direction, geometric_normal, cone_width);
		albedo = material.albedo(hit_tex_coord.x, hit_tex_coord.y, lod);
	}

	float curvature = triangle_get_curvature(
		hit_triangle.position_edge_1,
		hit_triangle.position_edge_2,
		hit_triangle.normal_edge_1,
		hit_triangle.normal_edge_2
	);
	cone_angle += -2.0f * curvature * fabsf(cone_width) / dot(hit_normal, ray_direction); // Eq. 5 (Akenine-Möller 2021)
#else
	albedo = material.albedo(hit_tex_coord.x, hit_tex_coord.y);
#endif

	if (bounce == 0 && (settings.demodulate_albedo || settings.enable_svgf)) {
		frame_buffer_albedo[ray_pixel_index] = make_float4(albedo);
	}


	float3 throughput = ray_throughput * albedo;
	
	if (settings.enable_next_event_estimation) {
		bool scene_has_lights = light_total_count_inv < INFINITY; // 1 / light_count < INF means light_count > 0
		if (scene_has_lights) {
			// Trace Shadow Ray
			float light_u, light_v;
			int   light_transform_id;
			int   light_id = random_point_on_random_light(x, y, sample_index, bounce, seed, light_u, light_v, light_transform_id);

			// Obtain the Light's position and normal
			TrianglePosNor light = triangle_get_positions_and_normals(light_id);

			float3 light_point;
			float3 light_normal;
			triangle_barycentric(light, light_u, light_v, light_point, light_normal);

			// Transform into world space
			Matrix3x4 light_world = mesh_get_transform(light_transform_id);
			matrix3x4_transform_position (light_world, light_point);
			matrix3x4_transform_direction(light_world, light_normal);

			float3 to_light = light_point - hit_point;
			float distance_to_light_squared = dot(to_light, to_light);
			float distance_to_light         = sqrtf(distance_to_light_squared);

			// Normalize the vector to the light
			to_light /= distance_to_light;

			float cos_o = -dot(to_light, light_normal);
			float cos_i =  dot(to_light,   hit_normal);
		
			// Only trace Shadow Ray if light transport is possible given the normals
			if (cos_o > 0.0f && cos_i > 0.0f) {
				// NOTE: N dot L is included here
				float brdf     = cos_i * ONE_OVER_PI;
				float brdf_pdf = cos_i * ONE_OVER_PI;

				float light_area = 0.5f * length(cross(light.position_edge_1, light.position_edge_2));

				float light_select_pdf = light_area / light_total_area;
				float light_pdf = light_select_pdf * distance_to_light_squared / (cos_o * light_area); // Convert solid angle measure

				float mis_pdf = settings.enable_multiple_importance_sampling ? brdf_pdf + light_pdf : light_pdf;

				float3 emission     = materials[triangle_get_material_id(light_id)].emission;
				float3 illumination = throughput * brdf * emission / mis_pdf;

				int shadow_ray_index = atomic_agg_inc(&buffer_sizes.shadow[bounce]);

				ray_buffer_shadow.ray_origin   .set(shadow_ray_index, hit_point);
				ray_buffer_shadow.ray_direction.set(shadow_ray_index, to_light);

				ray_buffer_shadow.max_distance[shadow_ray_index] = distance_to_light - EPSILON;

				ray_buffer_shadow.pixel_index[shadow_ray_index] = ray_pixel_index;
				ray_buffer_shadow.illumination.set(shadow_ray_index, illumination);
			}
		}
	}

	if (bounce == settings.num_bounces - 1) return;

	int index_out = atomic_agg_inc(&buffer_sizes.trace[bounce + 1]);

	float3 tangent, binormal; orthonormal_basis(hit_normal, tangent, binormal);

	float3 direction_local = random_cosine_weighted_direction(x, y, sample_index, bounce, seed);
	float3 direction_world = local_to_world(direction_local, tangent, binormal, hit_normal);

	ray_buffer_trace.origin   .set(index_out, hit_point);
	ray_buffer_trace.direction.set(index_out, direction_world);
	
#if ENABLE_MIPMAPPING
	ray_buffer_trace.cone[index_out] = make_float2(cone_angle, cone_width);
#endif

	ray_buffer_trace.pixel_index_and_last_material[index_out] = ray_pixel_index | int(Material::Type::DIFFUSE) << 30;
	ray_buffer_trace.throughput.set(index_out, throughput);

	ray_buffer_trace.last_pdf[index_out] = fabsf(dot(direction_world, hit_normal)) * ONE_OVER_PI;
}

extern "C" __global__ void kernel_shade_dielectric(int rand_seed, int bounce) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.dielectric[bounce] || bounce == settings.num_bounces - 1) return;

	float3 ray_direction = ray_buffer_shade_dielectric.direction.get(index);
	RayHit hit           = ray_buffer_shade_dielectric.hits     .get(index);

	int    ray_pixel_index = ray_buffer_shade_dielectric.pixel_index[index];
	float3 ray_throughput  = ray_buffer_shade_dielectric.throughput.get(index);

	ASSERT(hit.triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = wang_hash(index ^ rand_seed);

	const Material & material = materials[triangle_get_material_id(hit.triangle_id)];

	ASSERT(material.type == Material::Type::DIELECTRIC, "Material should be dielectric in this Kernel");

	// Obtain hit Triangle position, normal, and texture coordinates
	TrianglePosNor hit_triangle = triangle_get_positions_and_normals(hit.triangle_id);

	float3 hit_point;
	float3 hit_normal;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, hit_normal);

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, hit_normal);

	int index_out = atomic_agg_inc(&buffer_sizes.trace[bounce + 1]);

	// Calculate proper facing normal and determine indices of refraction
	float3 normal;
	float  cos_theta;

	float eta_1;
	float eta_2;

	float dir_dot_normal = dot(ray_direction, hit_normal);
	if (dir_dot_normal < 0.0f) { 
		// Entering material		
		eta_1 = 1.0f;
		eta_2 = material.index_of_refraction;

		normal    =  hit_normal;
		cos_theta = -dir_dot_normal;
	} else { 
		// Leaving material
		eta_1 = material.index_of_refraction;
		eta_2 = 1.0f;

		normal    = -hit_normal;
		cos_theta =  dir_dot_normal;

		// Lambert-Beer Law
		// NOTE: does not take into account nested dielectrics!
		if (dot(material.absorption, material.absorption) > EPSILON) {
			ray_throughput.x *= expf(material.absorption.x * hit.t);
			ray_throughput.y *= expf(material.absorption.y * hit.t);
			ray_throughput.z *= expf(material.absorption.z * hit.t);
		}
	}

	float eta = eta_1 / eta_2;
	float k = 1.0f - eta*eta * (1.0f - cos_theta*cos_theta);

	float3 ray_direction_reflected = reflect(ray_direction, hit_normal);
	float3 direction_out;

	if (k < 0.0f) { // Total Internal Reflection
		direction_out = ray_direction_reflected;
	} else {
		float3 ray_direction_refracted = normalize(eta * ray_direction + (eta * cos_theta - sqrtf(k)) * hit_normal);

		float f = fresnel(eta_1, eta_2, cos_theta, -dot(ray_direction_refracted, normal));

		if (random_float_xorshift(seed) < f) {
			direction_out = ray_direction_reflected;
		} else {
			direction_out = ray_direction_refracted;
		}
	}

	if (bounce == 0 && (settings.demodulate_albedo || settings.enable_svgf)) {
		frame_buffer_albedo[ray_pixel_index] = make_float4(1.0f);
	}

	ray_buffer_trace.origin   .set(index_out, hit_point);
	ray_buffer_trace.direction.set(index_out, direction_out);

#if ENABLE_MIPMAPPING
	float2 cone = ray_buffer_shade_dielectric.cone[index];
	float  cone_angle = cone.x;
	float  cone_width = cone.y + cone_angle * hit.t;
	
	float curvature = triangle_get_curvature(
		hit_triangle.position_edge_1,
		hit_triangle.position_edge_2,
		hit_triangle.normal_edge_1,
		hit_triangle.normal_edge_2
	);
	
	cone_angle += -2.0f * curvature * fabsf(cone_width) / dot(hit_normal, ray_direction); // Eq. 5 (Akenine-Möller 2021)

	ray_buffer_trace.cone[index_out] = make_float2(cone_angle, cone_width);
#endif
	ray_buffer_trace.pixel_index_and_last_material[index_out] = ray_pixel_index | int(Material::Type::DIELECTRIC) << 30;
	ray_buffer_trace.throughput.set(index_out, ray_throughput);
}

extern "C" __global__ void kernel_shade_glossy(int rand_seed, int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.glossy[bounce]) return;

	float3 ray_direction = ray_buffer_shade_glossy.direction.get(index);

	RayHit hit = ray_buffer_shade_glossy.hits.get(index);

	int ray_pixel_index = ray_buffer_shade_glossy.pixel_index[index];
	int x = ray_pixel_index % screen_pitch;
	int y = ray_pixel_index / screen_pitch; 

	float3 ray_throughput = ray_buffer_shade_glossy.throughput.get(index);

	ASSERT(hit.triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	unsigned seed = wang_hash(index ^ rand_seed);

	const Material & material = materials[triangle_get_material_id(hit.triangle_id)];

	ASSERT(material.type == Material::Type::GLOSSY, "Material should be glossy in this Kernel");

	// Obtain hit Triangle position, normal, and texture coordinates
	TrianglePosNorTex hit_triangle = triangle_get_positions_normals_and_tex_coords(hit.triangle_id);

	float3 hit_point;
	float3 hit_normal;
	float2 hit_tex_coord;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, hit_normal, hit_tex_coord);

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, hit_normal);

	if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	// Slightly widen the distribution to prevent the weights from becoming too large (see Walter et al. 2007)
	float alpha = (1.2f - 0.2f * sqrtf(-dot(ray_direction, hit_normal))) * material.roughness;
	
	float3 albedo;

#if ENABLE_MIPMAPPING
	float cone_angle, cone_width;

	float3 geometric_normal = normalize(cross(hit_triangle.position_edge_1, hit_triangle.position_edge_2));

	if (bounce == 0) {
		cone_angle = camera.pixel_spread_angle;
		cone_width = cone_angle * hit.t;

		float3 ellipse_axis_1, ellipse_axis_2; ray_cone_get_ellipse_axes(ray_direction, geometric_normal, cone_width, ellipse_axis_1, ellipse_axis_2);

		float2 gradient_1, gradient_2; ray_cone_get_texture_gradients(
			geometric_normal,
			hit_triangle.position_0,  hit_triangle.position_edge_1,  hit_triangle.position_edge_2,
			hit_triangle.tex_coord_0, hit_triangle.tex_coord_edge_1, hit_triangle.tex_coord_edge_2,
			hit_point, hit_tex_coord,
			ellipse_axis_1, ellipse_axis_2,
			gradient_1, gradient_2
		);

		// Anisotropic sampling
		albedo = material.albedo(hit_tex_coord.x, hit_tex_coord.y, gradient_1, gradient_2);
	} else {
		float2 cone = ray_buffer_shade_glossy.cone[index];
		cone_angle = cone.x;
		cone_width = cone.y + cone_angle * hit.t;

		// Trilinear sampling
		float lod = triangle_get_lod(hit.triangle_id) + ray_cone_get_lod(ray_direction, geometric_normal, cone_width);
		albedo = material.albedo(hit_tex_coord.x, hit_tex_coord.y, lod);
	}

	float curvature = triangle_get_curvature(
		hit_triangle.position_edge_1,
		hit_triangle.position_edge_2,
		hit_triangle.normal_edge_1,
		hit_triangle.normal_edge_2
	);

	cone_angle += -2.0f * curvature * fabsf(cone_width) / dot(hit_normal, ray_direction); // Eq. 5 (Akenine-Möller 2021)

	// Increase angle to account for roughness
//	float sigma_squared = 0.5f * (alpha * alpha / (1.0f - alpha * alpha)); // See Appendix A. of Akenine-Möller 2021
//	cone_angle += bounce == 0 ? 0.25f * sigma_squared : sigma_squared;
#else
	albedo = material.albedo(hit_tex_coord.x, hit_tex_coord.y);
#endif

	if (bounce == 0 && (settings.demodulate_albedo || settings.enable_svgf)) {
		frame_buffer_albedo[ray_pixel_index] = make_float4(albedo);
	}

	float3 throughput = ray_throughput * albedo;

	if (settings.enable_next_event_estimation) {
		bool scene_has_lights = light_total_count_inv < INFINITY; // 1 / light_count < INF means light_count > 0
		if (scene_has_lights && material.roughness >= ROUGHNESS_CUTOFF) {
			// Trace Shadow Ray
			float light_u;
			float light_v;
			int   light_transform_id;
			int   light_id = random_point_on_random_light(x, y, sample_index, bounce, seed, light_u, light_v, light_transform_id);

			// Obtain the Light's position and normal
			TrianglePosNor light = triangle_get_positions_and_normals(light_id);

			float3 light_point;
			float3 light_normal;
			triangle_barycentric(light, light_u, light_v, light_point, light_normal);

			// Transform into world space
			Matrix3x4 light_world = mesh_get_transform(light_transform_id);
			matrix3x4_transform_position (light_world, light_point);
			matrix3x4_transform_direction(light_world, light_normal);

			float3 to_light = light_point - hit_point;
			float distance_to_light_squared = dot(to_light, to_light);
			float distance_to_light         = sqrtf(distance_to_light_squared);

			// Normalize the vector to the light
			to_light /= distance_to_light;

			float cos_o = -dot(to_light, light_normal);
			float cos_i =  dot(to_light,   hit_normal);

			// Only trace Shadow Ray if light transport is possible given the normals
			if (cos_o > 0.0f && cos_i > 0.0f) {
				float3 half_vector = normalize(to_light - ray_direction);

				float i_dot_n = -dot(ray_direction, hit_normal);
				float m_dot_n =  dot(half_vector,   hit_normal);

				float F = fresnel(material.index_of_refraction, 1.0f, i_dot_n, i_dot_n);
				float D = microfacet_D(m_dot_n, alpha);
				float G = microfacet_G(i_dot_n, cos_i, i_dot_n, cos_i, m_dot_n, alpha);

				// NOTE: N dot L is omitted from the denominator here
				float brdf     = (F * G * D) / (4.0f * i_dot_n);
				float brdf_pdf = F * D * m_dot_n / (-4.0f * dot(half_vector, ray_direction));
				
				float light_area = 0.5f * length(cross(light.position_edge_1, light.position_edge_2));

				float light_select_pdf = light_area / light_total_area;
				float light_pdf = light_select_pdf * distance_to_light_squared / (cos_o * light_area); // Convert solid angle measure

				float mis_pdf = settings.enable_multiple_importance_sampling ? brdf_pdf + light_pdf : light_pdf;

				float3 emission     = materials[triangle_get_material_id(light_id)].emission;
				float3 illumination = throughput * brdf * emission / mis_pdf;

				int shadow_ray_index = atomic_agg_inc(&buffer_sizes.shadow[bounce]);

				ray_buffer_shadow.ray_origin   .set(shadow_ray_index, hit_point);
				ray_buffer_shadow.ray_direction.set(shadow_ray_index, to_light);

				ray_buffer_shadow.max_distance[shadow_ray_index] = distance_to_light - EPSILON;

				ray_buffer_shadow.pixel_index[shadow_ray_index] = ray_pixel_index;
				ray_buffer_shadow.illumination.set(shadow_ray_index, illumination);
			}
		}
	}

	if (bounce == settings.num_bounces - 1) return;

	// Sample normal distribution in spherical coordinates
	float theta = atanf(sqrtf(-alpha * alpha * logf(random_float_heitz(x, y, sample_index, bounce, 2, seed) + 1e-8f)));
	float phi   = TWO_PI * random_float_heitz(x, y, sample_index, bounce, 3, seed);

	float sin_theta, cos_theta; sincos(theta, &sin_theta, &cos_theta);
	float sin_phi,   cos_phi;   sincos(phi,   &sin_phi,   &cos_phi);

	// Convert from spherical coordinates to cartesian coordinates
	float3 micro_normal_local = make_float3(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);

	// Convert from tangent to world space
	float3 hit_tangent, hit_binormal;
	orthonormal_basis(hit_normal, hit_tangent, hit_binormal);

	float3 micro_normal_world = local_to_world(micro_normal_local, hit_tangent, hit_binormal, hit_normal);

	// Apply perfect mirror reflection to world space normal
	float3 direction_out = reflect(ray_direction, micro_normal_world);

	float i_dot_m = -dot(ray_direction, micro_normal_world);
	float o_dot_m =  dot(direction_out, micro_normal_world);
	float i_dot_n = -dot(ray_direction,      hit_normal);
	float o_dot_n =  dot(direction_out,      hit_normal);
	float m_dot_n =  dot(micro_normal_world, hit_normal);

	float F = fresnel(material.index_of_refraction, 1.0f, i_dot_m, i_dot_m);
	float D = microfacet_D(m_dot_n, alpha);
	float G = microfacet_G(i_dot_m, o_dot_m, i_dot_n, o_dot_n, m_dot_n, alpha);
	float weight = fabsf(i_dot_m) * F * G / fabsf(i_dot_n * m_dot_n);

	int index_out = atomic_agg_inc(&buffer_sizes.trace[bounce + 1]);

	ray_buffer_trace.origin   .set(index_out, hit_point);
	ray_buffer_trace.direction.set(index_out, direction_out);

	ray_buffer_trace.pixel_index_and_last_material[index_out] = ray_pixel_index | int(Material::Type::GLOSSY) << 30;
	ray_buffer_trace.throughput.set(index_out, throughput);

	ray_buffer_trace.last_pdf[index_out] = weight;
}

extern "C" __global__ void kernel_trace_shadow(int bounce) {
	bvh_trace_shadow(buffer_sizes.shadow[bounce], &buffer_sizes.rays_retired_shadow[bounce], bounce);
}

extern "C" __global__ void kernel_accumulate(float frames_accumulated) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float4 direct   = frame_buffer_direct  [pixel_index];
	float4 indirect = frame_buffer_indirect[pixel_index];

	float4 colour = direct + indirect;

	if (settings.demodulate_albedo) {
		colour /= fmaxf(frame_buffer_albedo[pixel_index], make_float4(1e-8f));
	}	

	if (frames_accumulated > 0.0f) {
		float4 colour_prev = accumulator.get(x, y);

		colour = colour_prev + (colour - colour_prev) / frames_accumulated; // Online average
	}

	accumulator.set(x, y, colour);

	// Clear frame buffers for next frame
	if (settings.demodulate_albedo) {
		frame_buffer_albedo[pixel_index] = make_float4(0.0f);
	}
	frame_buffer_direct  [pixel_index] = make_float4(0.0f);
	frame_buffer_indirect[pixel_index] = make_float4(0.0f);
}
