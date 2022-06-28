#include "cudart/vector_types.h"
#include "cudart/cuda_math.h"

#include "Common.h"

#include "Util.h"
#include "Buffers.h"
#include "BSDF.h"
#include "Material.h"
#include "Medium.h"
#include "Sky.h"
#include "RayCone.h"

#include "Raytracing/BVH2.h"
#include "Raytracing/BVH4.h"
#include "Raytracing/BVH8.h"

#include "Sampling.h"
#include "Camera.h"

#include "SVGF/SVGF.h"
#include "SVGF/TAA.h"

// Final Frame Buffer, shared with OpenGL
__device__ __constant__ Surface<float4> accumulator;

constexpr unsigned FLAG_ALLOW_NEE     = 1u << 31; // Indicates the previous Material has a BRDF that supports NEE
constexpr unsigned FLAG_INSIDE_MEDIUM = 1u << 30;

constexpr unsigned FLAGS_ALL = FLAG_ALLOW_NEE | FLAG_INSIDE_MEDIUM;

// Input to the Trace and Sort Kernels in SoA layout
struct TraceBuffer {
	TraversalData traversal_data;

	float * cone_angle;
	float * cone_width;

	int * medium;

	unsigned  * pixel_index_and_flags;
	Vector3_SoA throughput;

	float * last_pdf;
};

// Input to the Material Kernels in SoA layout
struct MaterialBuffer {
	Vector3_SoA ray_direction;

	HitBuffer hits;

	float * cone_angle;
	float * cone_width;

	int * medium;

	int       * pixel_index_and_flags;
	Vector3_SoA throughput;
};

// Input to the Shadow Trace Kernels in SoA layout
struct ShadowRayBuffer {
	ShadowTraversalData traversal_data;

	float4 * illumination_and_pixel_index;
};

__device__ __constant__ TraceBuffer     ray_buffer_trace_0;
__device__ __constant__ TraceBuffer     ray_buffer_trace_1;
__device__ __constant__ ShadowRayBuffer ray_buffer_shadow;

using PackedMaterialBuffer = size_t;

__device__ __constant__ PackedMaterialBuffer material_buffer_diffuse;
__device__ __constant__ PackedMaterialBuffer material_buffer_plastic;
__device__ __constant__ PackedMaterialBuffer material_buffer_dielectric;
__device__ __constant__ PackedMaterialBuffer material_buffer_conductor;

struct MaterialBufferAllocation {
	MaterialBuffer * buffer;
	bool             reversed;
};

__device__ inline MaterialBufferAllocation get_material_buffer(PackedMaterialBuffer packed) {
	return MaterialBufferAllocation {
		reinterpret_cast<MaterialBuffer *>(packed & ~1),
		bool(packed & 1)
	};
}

__device__ inline TraceBuffer * get_ray_buffer_trace(int bounce) {
	if (bounce & 1) {
		return &ray_buffer_trace_1;
	} else {
		return &ray_buffer_trace_0;
	}
}

// Number of elements in each Buffer
// Sizes are stored for ALL bounces so we only have to reset these
// values back to 0 after every frame, instead of after every bounce
struct BufferSizes {
	int trace     [MAX_BOUNCES];
	int diffuse   [MAX_BOUNCES];
	int plastic   [MAX_BOUNCES];
	int dielectric[MAX_BOUNCES];
	int conductor [MAX_BOUNCES];
	int shadow    [MAX_BOUNCES];

	// Global counters for tracing kernels
	int rays_retired       [MAX_BOUNCES];
	int rays_retired_shadow[MAX_BOUNCES];
};

__device__ BufferSizes buffer_sizes;

__device__ __constant__ Camera camera;

__device__ PixelQuery pixel_query = { INVALID, INVALID, INVALID };

extern "C" __global__ void kernel_generate(int sample_index, int pixel_offset, int pixel_count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= pixel_count) return;

	int index_offset = index + pixel_offset;
	int x = index_offset % screen_width;
	int y = index_offset / screen_width;

	int pixel_index = x + y * screen_pitch;

	Ray ray = camera_generate_ray(pixel_index, sample_index, x, y, camera);

	TraceBuffer * ray_buffer_trace = get_ray_buffer_trace(0);

	ray_buffer_trace->traversal_data.ray_origin   .set(index, ray.origin);
	ray_buffer_trace->traversal_data.ray_direction.set(index, ray.direction);
	ray_buffer_trace->pixel_index_and_flags[index] = pixel_index;
}

extern "C" __global__ void kernel_trace_bvh2(int bounce) {
	bvh2_trace(&get_ray_buffer_trace(bounce)->traversal_data, buffer_sizes.trace[bounce], &buffer_sizes.rays_retired[bounce]);
}

extern "C" __global__ void kernel_trace_bvh4(int bounce) {
	bvh4_trace(&get_ray_buffer_trace(bounce)->traversal_data, buffer_sizes.trace[bounce], &buffer_sizes.rays_retired[bounce]);
}

extern "C" __global__ void kernel_trace_bvh8(int bounce) {
	bvh8_trace(&get_ray_buffer_trace(bounce)->traversal_data, buffer_sizes.trace[bounce], &buffer_sizes.rays_retired[bounce]);
}

extern "C" __global__ void kernel_trace_shadow_bvh2(int bounce) {
	bvh2_trace_shadow(&ray_buffer_shadow.traversal_data, buffer_sizes.shadow[bounce], &buffer_sizes.rays_retired_shadow[bounce], [bounce](int ray_index) {
		float4 illumination_and_pixel_index = ray_buffer_shadow.illumination_and_pixel_index[ray_index];
		float3 illumination = make_float3(illumination_and_pixel_index);
		int    pixel_index  = __float_as_int(illumination_and_pixel_index.w);

		aov_framebuffer_add(AOVType::RADIANCE, pixel_index, make_float4(illumination));
		if (bounce == 0) {
			aov_framebuffer_set(AOVType::RADIANCE_DIRECT, pixel_index, make_float4(illumination));
		} else {
			aov_framebuffer_add(AOVType::RADIANCE_INDIRECT, pixel_index, make_float4(illumination));
		}
	});
}

extern "C" __global__ void kernel_trace_shadow_bvh4(int bounce) {
	bvh4_trace_shadow(&ray_buffer_shadow.traversal_data, buffer_sizes.shadow[bounce], &buffer_sizes.rays_retired_shadow[bounce], [bounce](int ray_index) {
		float4 illumination_and_pixel_index = ray_buffer_shadow.illumination_and_pixel_index[ray_index];
		float3 illumination = make_float3(illumination_and_pixel_index);
		int    pixel_index  = __float_as_int(illumination_and_pixel_index.w);

		aov_framebuffer_add(AOVType::RADIANCE, pixel_index, make_float4(illumination));
		if (bounce == 0) {
			aov_framebuffer_set(AOVType::RADIANCE_DIRECT, pixel_index, make_float4(illumination));
		} else {
			aov_framebuffer_add(AOVType::RADIANCE_INDIRECT, pixel_index, make_float4(illumination));
		}
	});
}

extern "C" __global__ void kernel_trace_shadow_bvh8(int bounce) {
	bvh8_trace_shadow(&ray_buffer_shadow.traversal_data, buffer_sizes.shadow[bounce], &buffer_sizes.rays_retired_shadow[bounce], [bounce](int ray_index) {
		float4 illumination_and_pixel_index = ray_buffer_shadow.illumination_and_pixel_index[ray_index];
		float3 illumination = make_float3(illumination_and_pixel_index);
		int    pixel_index  = __float_as_int(illumination_and_pixel_index.w);

		aov_framebuffer_add(AOVType::RADIANCE, pixel_index, make_float4(illumination));
		if (bounce == 0) {
			aov_framebuffer_set(AOVType::RADIANCE_DIRECT, pixel_index, make_float4(illumination));
		} else {
			aov_framebuffer_add(AOVType::RADIANCE_INDIRECT, pixel_index, make_float4(illumination));
		}
	});
}

// Returns true if the path should terminate
__device__ bool russian_roulette(int pixel_index, int bounce, int sample_index, float3 & throughput) {
	if (bounce == config.num_bounces - 1) {
		return true;
	}
	if (config.enable_russian_roulette && bounce > 0) {
		float3 throughput_with_albedo = throughput;
		if (config.enable_svgf) {
			throughput_with_albedo *= make_float3(aov_framebuffer_get(AOVType::ALBEDO, pixel_index));
		}

		float survival_probability  = __saturatef(vmax_max(throughput_with_albedo.x, throughput_with_albedo.y, throughput_with_albedo.z));
		float rand_russian_roulette = random<SampleDimension::RUSSIAN_ROULETTE>(pixel_index, bounce, sample_index).x;

		if (rand_russian_roulette > survival_probability) {
			return true;
		}
		throughput /= survival_probability;
	}
	return false;
}

extern "C" __global__ void kernel_sort(int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.trace[bounce]) return;

	const TraceBuffer * ray_buffer_trace = get_ray_buffer_trace(bounce);

	float3 ray_direction = ray_buffer_trace->traversal_data.ray_direction.get(index);
	RayHit hit           = ray_buffer_trace->traversal_data.hits         .get(index);

	float ray_cone_angle;
	float ray_cone_width;
	if (bounce > 0 && config.enable_mipmapping) {
		ray_cone_angle = ray_buffer_trace->cone_angle[index];
		ray_cone_width = ray_buffer_trace->cone_width[index];
	}

	unsigned pixel_index_and_flags = ray_buffer_trace->pixel_index_and_flags[index];
	int      pixel_index = pixel_index_and_flags & ~FLAGS_ALL;

	int x = pixel_index % screen_pitch;
	int y = pixel_index / screen_pitch;

	bool allow_nee     = pixel_index_and_flags & FLAG_ALLOW_NEE;
	bool inside_medium = pixel_index_and_flags & FLAG_INSIDE_MEDIUM;

	float3 throughput;
	if (bounce == 0) {
		throughput = make_float3(1.0f); // Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		throughput = ray_buffer_trace->throughput.get(index);
	}

	int medium_id = INVALID;
	if (inside_medium) {
		medium_id = ray_buffer_trace->medium[index];
		HomogeneousMedium medium = medium_as_homogeneous(medium_id);

		bool medium_can_scatter = (medium.sigma_s.x + medium.sigma_s.y + medium.sigma_s.z) > 0.0f;

		if (medium_can_scatter) {
			float2 rand_scatter = random<SampleDimension::BSDF_0>(pixel_index, bounce, sample_index);
			float2 rand_phase   = random<SampleDimension::BSDF_1>(pixel_index, bounce, sample_index);

			float3 sigma_t = medium.sigma_a + medium.sigma_s;

			// MIS based on throughput
			// See Wrenninge - Path Traced Subsurface Scattering using Anisotropic Phase Functions and Non-Exponential Free Flights
			float  throughput_sum = throughput.x + throughput.y + throughput.z;
			float3 wavelength_pdf = throughput / throughput_sum; // pdfs for choosing any of the 3 RGB wavelengths

			float sigma_t_used_for_sampling;
			if (rand_scatter.x * throughput_sum < throughput.x) {
				sigma_t_used_for_sampling = sigma_t.x;
			} else if (rand_scatter.x * throughput_sum < throughput.x + throughput.y) {
				sigma_t_used_for_sampling = sigma_t.y;
			} else {
				sigma_t_used_for_sampling = sigma_t.z;
			}

			float scatter_distance = sample_exp(sigma_t_used_for_sampling, rand_scatter.y);
			float3 transmittance = beer_lambert(sigma_t, fminf(scatter_distance, hit.t));

			if (scatter_distance < hit.t) {
				float3 pdf = wavelength_pdf * sigma_t * transmittance;
				throughput *= medium.sigma_s * transmittance / (pdf.x + pdf.y + pdf.z);

				if (russian_roulette(pixel_index, bounce, sample_index, throughput)) return;

				float3 direction_out = sample_henyey_greenstein(-ray_direction, medium.g, rand_phase.x, rand_phase.y);

				float3 ray_origin = ray_buffer_trace->traversal_data.ray_origin.get(index);
				float3 origin_out = ray_origin + scatter_distance * ray_direction;

				// Emit scattered Ray
				int index_out = atomicAdd(&buffer_sizes.trace[bounce + 1], 1);

				TraceBuffer * ray_buffer_trace_next = get_ray_buffer_trace(bounce + 1);

				ray_buffer_trace_next->traversal_data.ray_origin   .set(index_out, origin_out);
				ray_buffer_trace_next->traversal_data.ray_direction.set(index_out, direction_out);

				ray_buffer_trace_next->medium[index_out] = medium_id;

				if (config.enable_mipmapping) {
					if (bounce == 0) {
						// Ray Cone is normally initialized on the first bounce in the Material kernel.
						// Since a scattered Ray does not invoke a Material kernel, initialize the Ray Cone here
						ray_cone_angle = camera.pixel_spread_angle;
						ray_cone_width = camera.pixel_spread_angle * scatter_distance;
					}
					ray_buffer_trace_next->cone_angle[index_out] = ray_cone_angle;
					ray_buffer_trace_next->cone_width[index_out] = ray_cone_width;
				}

				ray_buffer_trace_next->pixel_index_and_flags[index_out] = pixel_index | FLAG_INSIDE_MEDIUM;
				ray_buffer_trace_next->throughput.set(index_out, throughput);

				return;
			} else {
				float3 pdf = wavelength_pdf * transmittance;
				throughput *= transmittance / (pdf.x + pdf.y + pdf.z);
			}
		} else {
			throughput *= beer_lambert(medium.sigma_a, hit.t);
		}
	}

	// If we didn't hit anything, sample the Sky
	if (hit.triangle_id == INVALID) {
		float3 illumination = throughput * sample_sky(ray_direction);

		if (bounce == 0) {
			aov_framebuffer_set(AOVType::ALBEDO,          pixel_index, make_float4(1.0f));
			aov_framebuffer_set(AOVType::RADIANCE,        pixel_index, make_float4(illumination));
			aov_framebuffer_set(AOVType::RADIANCE_DIRECT, pixel_index, make_float4(illumination));
		} else if (bounce == 1) {
			aov_framebuffer_add(AOVType::RADIANCE,        pixel_index, make_float4(illumination));
			aov_framebuffer_add(AOVType::RADIANCE_DIRECT, pixel_index, make_float4(illumination));
		} else {
			aov_framebuffer_add(AOVType::RADIANCE,          pixel_index, make_float4(illumination));
			aov_framebuffer_add(AOVType::RADIANCE_INDIRECT, pixel_index, make_float4(illumination));
		}
		return;
	}

	if (bounce == 0 && pixel_query.pixel_index == pixel_index) {
		pixel_query.mesh_id     = hit.mesh_id;
		pixel_query.triangle_id = hit.triangle_id;
	}

	// Get the Material of the Mesh we hit
	int material_id = mesh_get_material_id(hit.mesh_id);
	MaterialType material_type = material_get_type(material_id);

	if (material_type == MaterialType::LIGHT) {
		// Obtain the Light's position and normal
		TrianglePos light_triangle = triangle_get_positions(hit.triangle_id);

		float3 light_point;
		triangle_barycentric(light_triangle, hit.u, hit.v, light_point);

		float3 light_point_prev = light_point;

		float3 light_geometric_normal = cross(light_triangle.position_edge_1, light_triangle.position_edge_2);

		// Transform into world space
		Matrix3x4 light_world = mesh_get_transform(hit.mesh_id);
		matrix3x4_transform_position (light_world, light_point);
		matrix3x4_transform_direction(light_world, light_geometric_normal);
	
		light_geometric_normal = normalize(light_geometric_normal);
	
		if (bounce == 0 && config.enable_svgf) {
			Matrix3x4 world_prev = mesh_get_transform_prev(hit.mesh_id);
			matrix3x4_transform_position(world_prev, light_point_prev);

			svgf_set_gbuffers(x, y, hit, light_point, light_geometric_normal, light_point_prev);
		}

		MaterialLight material_light = material_as_light(material_id);

		bool should_count_light_contribution = config.enable_next_event_estimation ? !allow_nee : true;
		if (should_count_light_contribution) {
			float3 illumination = throughput * material_light.emission;

			if (bounce == 0) {
				aov_framebuffer_set(AOVType::ALBEDO,          pixel_index, make_float4(1.0f));
				aov_framebuffer_set(AOVType::RADIANCE,        pixel_index, make_float4(material_light.emission));
				aov_framebuffer_set(AOVType::RADIANCE_DIRECT, pixel_index, make_float4(material_light.emission));
			} else if (bounce == 1) {
				aov_framebuffer_add(AOVType::RADIANCE,        pixel_index, make_float4(illumination));
				aov_framebuffer_add(AOVType::RADIANCE_DIRECT, pixel_index, make_float4(illumination));
			} else {
				aov_framebuffer_add(AOVType::RADIANCE,          pixel_index, make_float4(illumination));
				aov_framebuffer_add(AOVType::RADIANCE_INDIRECT, pixel_index, make_float4(illumination));
			}
			return;
		}

		if (config.enable_multiple_importance_sampling) {
			float cos_theta_light = abs_dot(ray_direction, light_geometric_normal);
			float distance_to_light_squared = hit.t * hit.t;

			float brdf_pdf = ray_buffer_trace->last_pdf[index];

			float light_power = luminance(material_light.emission.x, material_light.emission.y, material_light.emission.z);
			float light_pdf   = light_power * distance_to_light_squared / (cos_theta_light * lights_total_weight);

			if (!pdf_is_valid(light_pdf)) return;

			float mis_weight = power_heuristic(brdf_pdf, light_pdf);
			float3 illumination = throughput * material_light.emission * mis_weight;

			assert(bounce != 0);
			aov_framebuffer_add(AOVType::RADIANCE, pixel_index, make_float4(illumination));
			if (bounce == 1) {
				aov_framebuffer_add(AOVType::RADIANCE_DIRECT, pixel_index, make_float4(illumination));
			} else {
				aov_framebuffer_add(AOVType::RADIANCE_INDIRECT, pixel_index, make_float4(illumination));
			}
		}
		return;
	}

	if (russian_roulette(pixel_index, bounce, sample_index, throughput)) return;

	unsigned flags = (medium_id != INVALID) << 30;

	auto material_buffer_write = [](
		int                  bounce,
		PackedMaterialBuffer packed_material_buffer,
		int                * buffer_size,
		const float3       & ray_direction,
		int                  medium_id,
		float                ray_cone_angle,
		float                ray_cone_width,
		const RayHit         hit,
		unsigned             pixel_index_and_flags,
		const float3       & throughput
	) {
		MaterialBufferAllocation material_buffer = get_material_buffer(packed_material_buffer);

		int index_out = atomicAdd(buffer_size, 1);
		if (material_buffer.reversed) {
			index_out = (BATCH_SIZE - 1) - index_out;
		}

		material_buffer.buffer->ray_direction.set(index_out, ray_direction);

		if (medium_id != INVALID) {
			material_buffer.buffer->medium[index_out] = medium_id;
		}

		if (bounce > 0 && config.enable_mipmapping) {
			material_buffer.buffer->cone_angle[index_out] = ray_cone_angle;
			material_buffer.buffer->cone_width[index_out] = ray_cone_width;
		}

		material_buffer.buffer->hits.set(index_out, hit);

		material_buffer.buffer->pixel_index_and_flags[index_out] = pixel_index_and_flags;
		if (bounce > 0) {
			material_buffer.buffer->throughput.set(index_out, throughput);
		}
	};

	switch (material_type) {
		case MaterialType::DIFFUSE: {
			material_buffer_write(bounce, material_buffer_diffuse, &buffer_sizes.diffuse[bounce], ray_direction, medium_id, ray_cone_angle, ray_cone_width, hit, pixel_index | flags, throughput);
			break;
		}
		case MaterialType::PLASTIC: {
			material_buffer_write(bounce, material_buffer_plastic, &buffer_sizes.plastic[bounce], ray_direction, medium_id, ray_cone_angle, ray_cone_width, hit, pixel_index | flags, throughput);
			break;
		}
		case MaterialType::DIELECTRIC: {
			material_buffer_write(bounce, material_buffer_dielectric, &buffer_sizes.dielectric[bounce], ray_direction, medium_id, ray_cone_angle, ray_cone_width, hit, pixel_index | flags, throughput);
			break;
		}
		case MaterialType::CONDUCTOR: {
			material_buffer_write(bounce, material_buffer_conductor, &buffer_sizes.conductor[bounce], ray_direction, medium_id, ray_cone_angle, ray_cone_width, hit, pixel_index | flags, throughput);
			break;
		}
	}
}

template<typename BSDF>
__device__ void next_event_estimation(
	int            pixel_index,
	int            bounce,
	int            sample_index,
	const BSDF   & bsdf,
	int            medium_id,
	const float3 & hit_point,
	const float3 & normal,
	const float3 & geometric_normal,
	float3       & throughput
) {
	float2 rand_light    = random<SampleDimension::NEE_LIGHT>   (pixel_index, bounce, sample_index);
	float2 rand_triangle = random<SampleDimension::NEE_TRIANGLE>(pixel_index, bounce, sample_index);

	// Pick random Light
	int light_mesh_id;
	int light_triangle_id = sample_light(rand_light.x, rand_light.y, light_mesh_id);

	// Pick random point on the Light
	float2 light_uv = sample_triangle(rand_triangle.x, rand_triangle.y);

	// Obtain the Light's position and geometric normal
	TrianglePos light_triangle = triangle_get_positions(light_triangle_id);

	float3 light_point;
	triangle_barycentric(light_triangle, light_uv.x, light_uv.y, light_point);

	float3 light_geometric_normal = cross(light_triangle.position_edge_1, light_triangle.position_edge_2);

	// Transform into world space
	Matrix3x4 light_world = mesh_get_transform(light_mesh_id);
	matrix3x4_transform_position (light_world, light_point);
	matrix3x4_transform_direction(light_world, light_geometric_normal);

	light_geometric_normal = normalize(light_geometric_normal);

	float3 to_light = light_point - hit_point;
	float distance_to_light = length(to_light);
	to_light /= distance_to_light;

	float cos_theta_light = abs_dot(to_light, light_geometric_normal);
	float cos_theta_hit = dot(to_light, normal);

	int light_material_id = mesh_get_material_id(light_mesh_id);
	MaterialLight material_light = material_as_light(light_material_id);

	float3 bsdf_value;
	float  bsdf_pdf;
	bool valid = bsdf.eval(to_light, cos_theta_hit, bsdf_value, bsdf_pdf);
	if (!valid) return;

	float light_power = luminance(material_light.emission.x, material_light.emission.y, material_light.emission.z);
	float light_pdf   = light_power * square(distance_to_light) / (cos_theta_light * lights_total_weight);

	if (!pdf_is_valid(light_pdf)) return;

	float mis_weight;
	if (config.enable_multiple_importance_sampling) {
		mis_weight = power_heuristic(light_pdf, bsdf_pdf);
	} else {
		mis_weight = 1.0f;
	}

	float3 illumination = throughput * bsdf_value * material_light.emission * mis_weight / light_pdf;

	// If inside a Medium, apply absorption and out-scattering
	if (medium_id != INVALID) {
		HomogeneousMedium medium = medium_as_homogeneous(medium_id);
		float3 sigma_t = medium.sigma_a + medium.sigma_s;
		illumination *= beer_lambert(sigma_t, distance_to_light);
	}

	// Emit Shadow Ray
	int shadow_ray_index = atomicAdd(&buffer_sizes.shadow[bounce], 1);

	ray_buffer_shadow.traversal_data.ray_origin   .set(shadow_ray_index, ray_origin_epsilon_offset(hit_point, to_light, geometric_normal));
	ray_buffer_shadow.traversal_data.ray_direction.set(shadow_ray_index, to_light);
	ray_buffer_shadow.traversal_data.max_distance[shadow_ray_index] = distance_to_light - 2.0f * EPSILON;
	ray_buffer_shadow.illumination_and_pixel_index[shadow_ray_index] = make_float4(
		illumination.x,
		illumination.y,
		illumination.z,
		__int_as_float(pixel_index)
	);
}

template<typename BSDF, PackedMaterialBuffer * packed_material_buffer>
__device__ void shade_material(int bounce, int sample_index, int buffer_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_size) return;

	MaterialBufferAllocation material_buffer = get_material_buffer(*packed_material_buffer);

	// Material Buffers can be shared by 2 different Materials, one growing left to right, one growing right to left
	// If this Material is right to left, reverse the index into the buffers
	if (material_buffer.reversed) {
		index = (BATCH_SIZE - 1) - index;
	}

	float3 ray_direction = material_buffer.buffer->ray_direction.get(index);
	RayHit hit           = material_buffer.buffer->hits         .get(index);

	unsigned pixel_index_and_flags = material_buffer.buffer->pixel_index_and_flags[index];
	int      pixel_index = pixel_index_and_flags & ~FLAGS_ALL;

	bool inside_medium = pixel_index_and_flags & FLAG_INSIDE_MEDIUM;

	int medium_id = INVALID;
	if (inside_medium) {
		medium_id = material_buffer.buffer->medium[index];
	}

	float3 throughput;
	if (bounce == 0) {
		throughput = make_float3(1.0f); // Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		throughput = material_buffer.buffer->throughput.get(index);
	}

	// Obtain hit Triangle position, normal, and texture coordinates
	TrianglePosNorTex hit_triangle = triangle_get_positions_normals_and_tex_coords(hit.triangle_id);

	float3 hit_point;
	float3 normal;
	float2 tex_coord;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, normal, tex_coord);

	float3 hit_point_local = hit_point; // Keep copy of the untransformed hit point in local space

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, normal);

	normal = normalize(normal);

	float mesh_scale_inv = 1.0f / mesh_get_scale(hit.mesh_id);

	// Load and propagate Ray Cone
	float cone_angle;
	float cone_width;
	float curvature = 0.0f;
	if (config.enable_mipmapping) {
		if (bounce == 0) {
			cone_angle = camera.pixel_spread_angle;
			cone_width = cone_angle * hit.t;
		} else {
			cone_angle = material_buffer.buffer->cone_angle[index];
			cone_width = material_buffer.buffer->cone_width[index] + cone_angle * hit.t;
		}

		// Calculate Triangle curvature here,
		// not yet needed (see below) but after this step the Triangle position edges are transformed into world space
		curvature = triangle_get_curvature(
			hit_triangle.position_edge_1,
			hit_triangle.position_edge_2,
			hit_triangle.normal_edge_1,
			hit_triangle.normal_edge_2
		) * mesh_scale_inv;
	}

	matrix3x4_transform_direction(world, hit_triangle.position_edge_1);
	matrix3x4_transform_direction(world, hit_triangle.position_edge_2);

	// Calculate geometric normal (in world space) to the Triangle
	float3 geometric_normal = cross(hit_triangle.position_edge_1, hit_triangle.position_edge_2);
	float triangle_double_area_inv = 1.0f / length(geometric_normal);
	geometric_normal *= triangle_double_area_inv; // Normalize

	// Check which side of the Triangle we are on based on its geometric normal
	bool entering_material = dot(ray_direction, geometric_normal) < 0.0f;
	if (!entering_material) {
		normal    = -normal;
		curvature = -curvature;
	}

	// Construct TBN frame
	float3 tangent, bitangent;
	orthonormal_basis(normal, tangent, bitangent);

	float3 omega_i = world_to_local(-ray_direction, tangent, bitangent, normal);

	if (omega_i.z <= 0.0f) return; // Below hemisphere, reject

	// Initialize BSDF
	int material_id = mesh_get_material_id(hit.mesh_id);

	BSDF bsdf;
	bsdf.pixel_index  = pixel_index;
	bsdf.bounce       = bounce;
	bsdf.sample_index = sample_index;
	bsdf.tangent      = tangent;
	bsdf.bitangent    = bitangent;
	bsdf.normal       = normal;
	bsdf.omega_i      = omega_i;
	bsdf.init(bounce, entering_material, material_id);

	if (BSDF::HAS_ALBEDO) {
		TextureLOD lod;

		if (config.enable_mipmapping && bsdf.has_texture()) {
			if (use_anisotropic_texture_sampling(bounce)) {
				float3 ellipse_axis_1;
				float3 ellipse_axis_2;
				ray_cone_get_ellipse_axes(ray_direction, geometric_normal, cone_width, ellipse_axis_1, ellipse_axis_2);

				lod.aniso.gradient_1 = ray_cone_ellipse_axis_to_gradient(hit_triangle, triangle_double_area_inv, geometric_normal, hit_point, tex_coord, ellipse_axis_1);
				lod.aniso.gradient_2 = ray_cone_ellipse_axis_to_gradient(hit_triangle, triangle_double_area_inv, geometric_normal, hit_point, tex_coord, ellipse_axis_2);
			} else {
				float lod_triangle = triangle_get_lod(triangle_double_area_inv, hit_triangle.tex_coord_edge_1, hit_triangle.tex_coord_edge_2);
				float lod_ray_cone = ray_cone_get_lod(ray_direction, geometric_normal, cone_width);

				lod.iso.lod = log2f(lod_triangle * lod_ray_cone);
			}
		}

		bsdf.calc_albedo(bounce, pixel_index, throughput, tex_coord, lod);
	} else if (bounce == 0) {
		aov_framebuffer_set(AOVType::ALBEDO, pixel_index, make_float4(1.0f));
	}

	if (bounce == 0) {
		aov_framebuffer_set(AOVType::NORMAL,   pixel_index, make_float4(normal));
		aov_framebuffer_set(AOVType::POSITION, pixel_index, make_float4(hit_point));
	}

	// Calulate new Ray Cone angle
	if (config.enable_mipmapping) {
		cone_angle -= 2.0f * curvature * fabsf(cone_width) / dot(normal, ray_direction); // Eq. 5 (Akenine-Möller 2021)
	}

	// Emit GBuffers if SVGF is enabled
	if (bounce == 0 && config.enable_svgf) {
		float3 hit_point_prev = hit_point_local;

		Matrix3x4 world_prev = mesh_get_transform_prev(hit.mesh_id);
		matrix3x4_transform_position(world_prev, hit_point_prev);

		int x = pixel_index % screen_pitch;
		int y = pixel_index / screen_pitch;
		svgf_set_gbuffers(x, y, hit, hit_point, normal, hit_point_prev);
	}

	// Next Event Estimation
	if (config.enable_next_event_estimation && lights_total_weight > 0.0f && bsdf.allow_nee()) {
		next_event_estimation(pixel_index, bounce, sample_index, bsdf, medium_id, hit_point, normal, geometric_normal, throughput);
	}

	// Sample BSDF
	float3 direction_out;
	float pdf;
	bool valid = bsdf.sample(throughput, medium_id, direction_out, pdf);

	if (!valid) return;

	float3 origin_out = ray_origin_epsilon_offset(hit_point, direction_out, geometric_normal);

	// Emit next Ray
	int index_out = atomicAdd(&buffer_sizes.trace[bounce + 1], 1);

	TraceBuffer * ray_buffer_trace = get_ray_buffer_trace(bounce + 1);

	ray_buffer_trace->traversal_data.ray_origin   .set(index_out, origin_out);
	ray_buffer_trace->traversal_data.ray_direction.set(index_out, direction_out);

	if (medium_id != INVALID) {
		ray_buffer_trace->medium[index_out] = medium_id;
	}

	if (config.enable_mipmapping) {
		ray_buffer_trace->cone_angle[index_out] = cone_angle;
		ray_buffer_trace->cone_width[index_out] = cone_width;
	}

	bool allow_nee = bsdf.allow_nee();

	unsigned flags = 0;
	if (allow_nee)            flags |= FLAG_ALLOW_NEE;
	if (medium_id != INVALID) flags |= FLAG_INSIDE_MEDIUM;

	ray_buffer_trace->pixel_index_and_flags[index_out] = pixel_index | flags;
	ray_buffer_trace->throughput.set(index_out, throughput);

	if (allow_nee) {
		ray_buffer_trace->last_pdf[index_out] = pdf;
	}
}

extern "C" __global__ void kernel_material_diffuse(int bounce, int sample_index) {
	shade_material<BSDFDiffuse, &material_buffer_diffuse>(bounce, sample_index, buffer_sizes.diffuse[bounce]);
}

extern "C" __global__ void kernel_material_plastic(int bounce, int sample_index) {
	shade_material<BSDFPlastic, &material_buffer_plastic>(bounce, sample_index, buffer_sizes.plastic[bounce]);
}

extern "C" __global__ void kernel_material_dielectric(int bounce, int sample_index) {
	shade_material<BSDFDielectric, &material_buffer_dielectric>(bounce, sample_index, buffer_sizes.dielectric[bounce]);
}

extern "C" __global__ void kernel_material_conductor(int bounce, int sample_index) {
	shade_material<BSDFConductor, &material_buffer_conductor>(bounce, sample_index, buffer_sizes.conductor[bounce]);
}

extern "C" __global__ void kernel_accumulate(float frames_accumulated) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float4 colour = aov_accumulate(AOVType::RADIANCE, pixel_index, frames_accumulated);

	// Accumulate auxilary AOVs (if present)
	aov_accumulate(AOVType::ALBEDO,   pixel_index, frames_accumulated);
	aov_accumulate(AOVType::NORMAL,   pixel_index, frames_accumulated);
	aov_accumulate(AOVType::POSITION, pixel_index, frames_accumulated);

	if (!isfinite(colour.x + colour.y + colour.z)) {
//		printf("WARNING: pixel (%i, %i) has colour (%f, %f, %f)!\n", x, y, colour.x, colour.y, colour.z);
		colour = make_float4(1000.0f, 0.0f, 1000.0f, 1.0f);
	}

	accumulator.set(x, y, colour);
}
