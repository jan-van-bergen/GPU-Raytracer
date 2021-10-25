#include "cudart/vector_types.h"
#include "cudart/cuda_math.h"

#include "Common.h"

#define INFINITY ((float)(1e+300 * 1e+300))

__device__ __constant__ int screen_width;
__device__ __constant__ int screen_pitch;
__device__ __constant__ int screen_height;

__device__ __constant__ Config config;

// Frame Buffers
__device__ __constant__ float4 * frame_buffer_albedo;
__device__ __constant__ float4 * frame_buffer_direct;
__device__ __constant__ float4 * frame_buffer_indirect;

#include "Util.h"
#include "BSDF.h"
#include "Material.h"
#include "Sky.h"
#include "RayCone.h"

// Final Frame Buffer, shared with OpenGL
__device__ __constant__ Surface<float4> accumulator;

#include "Raytracing/BVH.h"
#include "Raytracing/QBVH.h"
#include "Raytracing/CWBVH.h"

#include "Sampling.h"

#include "SVGF/SVGF.h"
#include "SVGF/TAA.h"

struct Camera {
	float3 position;
	float3 bottom_left_corner;
	float3 x_axis;
	float3 y_axis;
	float  pixel_spread_angle;
	float  aperture_radius;
	float  focal_distance;
} __device__ __constant__ camera;

__device__ PixelQuery pixel_query = { INVALID, INVALID, INVALID, INVALID };

extern "C" __global__ void kernel_generate(int sample_index, int pixel_offset, int pixel_count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= pixel_count) return;

	int index_offset = index + pixel_offset;
	int x = index_offset % screen_width;
	int y = index_offset / screen_width;

	int pixel_index = x + y * screen_pitch;
	ASSERT(pixel_index < screen_pitch * screen_height, "Pixel should fit inside the buffer");

	float2 rand_filter   = random<SampleDimension::FILTER>  (pixel_index, 0, sample_index);
	float2 rand_aperture = random<SampleDimension::APERTURE>(pixel_index, 0, sample_index);

	float2 jitter;

	if (config.enable_svgf) {
		jitter.x = taa_halton_x[sample_index & (TAA_HALTON_NUM_SAMPLES-1)];
		jitter.y = taa_halton_y[sample_index & (TAA_HALTON_NUM_SAMPLES-1)];
	} else {
		switch (config.reconstruction_filter) {
			case ReconstructionFilter::BOX: {
				jitter = rand_filter;
				break;
			}
			case ReconstructionFilter::TENT: {
				jitter.x = sample_tent(rand_filter.x);
				jitter.y = sample_tent(rand_filter.y);
				break;
			}
			case ReconstructionFilter::GAUSSIAN: {
				float2 gaussians = sample_gaussian(rand_filter.x, rand_filter.y);
				jitter.x = 0.5f + 0.5f * gaussians.x;
				jitter.y = 0.5f + 0.5f * gaussians.y;
				break;
			}
		}
	}

	float x_jittered = float(x) + jitter.x;
	float y_jittered = float(y) + jitter.y;

	float3 focal_point = camera.focal_distance * normalize(camera.bottom_left_corner + x_jittered * camera.x_axis + y_jittered * camera.y_axis);
	float2 lens_point  = camera.aperture_radius * sample_disk(rand_aperture.x, rand_aperture.y);

	float3 offset = camera.x_axis * lens_point.x + camera.y_axis * lens_point.y;
	float3 direction = normalize(focal_point - offset);

	// Create primary Ray that starts at the Camera's position and goes through the current pixel
	ray_buffer_trace.origin   .set(index, camera.position + offset);
	ray_buffer_trace.direction.set(index, direction);

	ray_buffer_trace.pixel_index_and_mis_eligable[index] = pixel_index | (false << 31);
}

extern "C" __global__ void kernel_trace_bvh(int bounce) {
	bvh_trace(buffer_sizes.trace[bounce], &buffer_sizes.rays_retired[bounce]);
}

extern "C" __global__ void kernel_trace_qbvh(int bounce) {
	qbvh_trace(buffer_sizes.trace[bounce], &buffer_sizes.rays_retired[bounce]);
}

extern "C" __global__ void kernel_trace_cwbvh(int bounce) {
	cwbvh_trace(buffer_sizes.trace[bounce], &buffer_sizes.rays_retired[bounce]);
}

extern "C" __global__ void kernel_trace_shadow_bvh(int bounce) {
	bvh_trace_shadow(buffer_sizes.shadow[bounce], &buffer_sizes.rays_retired_shadow[bounce], bounce);
}

extern "C" __global__ void kernel_trace_shadow_qbvh(int bounce) {
	qbvh_trace_shadow(buffer_sizes.shadow[bounce], &buffer_sizes.rays_retired_shadow[bounce], bounce);
}

extern "C" __global__ void kernel_trace_shadow_cwbvh(int bounce) {
	cwbvh_trace_shadow(buffer_sizes.shadow[bounce], &buffer_sizes.rays_retired_shadow[bounce], bounce);
}

extern "C" __global__ void kernel_sort(int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.trace[bounce]) return;

	float3 ray_direction = ray_buffer_trace.direction.get(index);

	RayHit hit = ray_buffer_trace.hits.get(index);

	unsigned pixel_index_and_mis_eligable = ray_buffer_trace.pixel_index_and_mis_eligable[index];
	int      pixel_index = pixel_index_and_mis_eligable & ~(0b11 << 31);

	int x = pixel_index % screen_pitch;
	int y = pixel_index / screen_pitch;

	bool mis_eligable = pixel_index_and_mis_eligable >> 31;

	float3 throughput;
	if (bounce == 0) {
		throughput = make_float3(1.0f); // Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		throughput = ray_buffer_trace.throughput.get(index);
	}

	// If we didn't hit anything, sample the Sky
	if (hit.triangle_id == INVALID) {
		float3 illumination = throughput * sample_sky(ray_direction);

		if (bounce == 0) {
			if (config.enable_albedo || config.enable_svgf) {
				frame_buffer_albedo[pixel_index] = make_float4(1.0f);
			}
			frame_buffer_direct[pixel_index] = make_float4(illumination);
		} else if (bounce == 1) {
			frame_buffer_direct[pixel_index] += make_float4(illumination);
		} else {
			frame_buffer_indirect[pixel_index] += make_float4(illumination);
		}

		return;
	}

	// Get the Material of the Mesh we hit
	int material_id = mesh_get_material_id(hit.mesh_id);
	MaterialType material_type = material_get_type(material_id);

	if (bounce == 0 && pixel_query.pixel_index == pixel_index) {
		pixel_query.mesh_id     = hit.mesh_id;
		pixel_query.triangle_id = hit.triangle_id;
		pixel_query.material_id = material_id;
	}

	if (material_type == MaterialType::LIGHT) {
		// Obtain the Light's position and normal
		TrianglePosNor light = triangle_get_positions_and_normals(hit.triangle_id);

		float3 light_point;
		float3 light_normal;
		triangle_barycentric(light, hit.u, hit.v, light_point, light_normal);

		float3 light_point_prev = light_point;

		// Transform into world space
		Matrix3x4 world = mesh_get_transform(hit.mesh_id);
		matrix3x4_transform_position (world, light_point);
		matrix3x4_transform_direction(world, light_normal);

		light_normal = normalize(light_normal);

		if (bounce == 0 && config.enable_svgf) {
			Matrix3x4 world_prev = mesh_get_transform_prev(hit.mesh_id);
			matrix3x4_transform_position(world_prev, light_point_prev);

			svgf_set_gbuffers(x, y, hit, light_point, light_normal, light_point_prev);
		}

		MaterialLight material_light = material_as_light(material_id);

		bool should_count_light_contribution = config.enable_next_event_estimation ? !mis_eligable : true;
		if (should_count_light_contribution) {
			float3 illumination = throughput * material_light.emission;

			if (bounce == 0) {
				if (config.enable_albedo || config.enable_svgf) {
					frame_buffer_albedo[pixel_index] = make_float4(1.0f);
				}
				frame_buffer_direct[pixel_index] = make_float4(material_light.emission);
			} else if (bounce == 1) {
				frame_buffer_direct[pixel_index] += make_float4(illumination);
			} else {
				frame_buffer_indirect[pixel_index] += make_float4(illumination);
			}

			return;
		}

		if (config.enable_multiple_importance_sampling) {
			float cos_theta_light = fabsf(dot(ray_direction, light_normal));
			float distance_to_light_squared = hit.t * hit.t;

			float brdf_pdf = ray_buffer_trace.last_pdf[index];

			float light_power = luminance(material_light.emission.x, material_light.emission.y, material_light.emission.z);
			float light_pdf   = light_power * distance_to_light_squared / (cos_theta_light * lights_total_weight);

			float mis_weight = power_heuristic(brdf_pdf, light_pdf);
			float3 illumination = throughput * material_light.emission * mis_weight;

			assert(bounce != 0);
			if (bounce == 1) {
				frame_buffer_direct[pixel_index] += make_float4(illumination);
			} else {
				frame_buffer_indirect[pixel_index] += make_float4(illumination);
			}
		}

		return;
	}

	// If this is the last bounce and we haven't hit a light, terminate
	if (bounce == config.num_bounces - 1) return;

	// Russian Roulette
	if (config.enable_russian_roulette && bounce > 0) {
		// Throughput does not include albedo so it doesn't need to be demodulated by SVGF (causing precision issues)
		// This deteriorates Russian Roulette performance, so albedo is included here
		float3 throughput_with_albedo = throughput * make_float3(frame_buffer_albedo[pixel_index]);

		float survival_probability  = saturate(vmax_max(throughput_with_albedo.x, throughput_with_albedo.y, throughput_with_albedo.z));
		float rand_russian_roulette = random<SampleDimension::RUSSIAN_ROULETTE>(pixel_index, bounce, sample_index).x;

		if (rand_russian_roulette > survival_probability) {
			return;
		}

		throughput /= survival_probability;
	}

	switch (material_type) {
		case MaterialType::DIFFUSE: {
			int index_out = atomic_agg_inc(&buffer_sizes.diffuse[bounce]);

			ray_buffer_shade_diffuse_and_plastic.direction.set(index_out, ray_direction);

			if (bounce > 0 && config.enable_mipmapping) ray_buffer_shade_diffuse_and_plastic.cone[index_out] = ray_buffer_trace.cone[index];

			ray_buffer_shade_diffuse_and_plastic.hits.set(index_out, hit);

			ray_buffer_shade_diffuse_and_plastic.pixel_index[index_out] = pixel_index;
			if (bounce > 0) ray_buffer_shade_diffuse_and_plastic.throughput.set(index_out, throughput);

			break;
		}

		case MaterialType::PLASTIC: {
			// Plastic Material buffer is shared with Diffuse Material buffer but grows in the opposite direction
			int index_out = (BATCH_SIZE - 1) - atomic_agg_inc(&buffer_sizes.plastic[bounce]);

			ray_buffer_shade_diffuse_and_plastic.direction.set(index_out, ray_direction);

			if (bounce > 0 && config.enable_mipmapping) ray_buffer_shade_diffuse_and_plastic.cone[index_out] = ray_buffer_trace.cone[index];

			ray_buffer_shade_diffuse_and_plastic.hits.set(index_out, hit);

			ray_buffer_shade_diffuse_and_plastic.pixel_index[index_out] = pixel_index;
			if (bounce > 0) ray_buffer_shade_diffuse_and_plastic.throughput.set(index_out, throughput);

			break;
		}

		case MaterialType::DIELECTRIC: {
			int index_out = atomic_agg_inc(&buffer_sizes.dielectric[bounce]);

			ray_buffer_shade_dielectric_and_conductor.direction.set(index_out, ray_direction);

			if (bounce > 0 && config.enable_mipmapping) ray_buffer_shade_dielectric_and_conductor.cone[index_out] = ray_buffer_trace.cone[index];

			ray_buffer_shade_dielectric_and_conductor.hits.set(index_out, hit);

			ray_buffer_shade_dielectric_and_conductor.pixel_index[index_out] = pixel_index;
			if (bounce > 0) ray_buffer_shade_dielectric_and_conductor.throughput.set(index_out, throughput);

			break;
		}

		case MaterialType::CONDUCTOR: {
			// Conductor Material buffer is shared with Dielectric Material buffer but grows in the opposite direction
			int index_out = (BATCH_SIZE - 1) - atomic_agg_inc(&buffer_sizes.conductor[bounce]);

			ray_buffer_shade_dielectric_and_conductor.direction.set(index_out, ray_direction);

			if (bounce > 0 && config.enable_mipmapping) ray_buffer_shade_dielectric_and_conductor.cone[index_out] = ray_buffer_trace.cone[index];

			ray_buffer_shade_dielectric_and_conductor.hits.set(index_out, hit);

			ray_buffer_shade_dielectric_and_conductor.pixel_index[index_out] = pixel_index;
			if (bounce > 0) ray_buffer_shade_dielectric_and_conductor.throughput.set(index_out, throughput);

			break;
		}
	}
}

template<typename BSDF, MaterialBuffer * material_buffer, bool REVERSED>
__device__ void shade_material(int bounce, int sample_index, int buffer_size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_size) return;

	// Material Buffers can be shared by 2 different Materials, one growing left to right, one growing right to left
	// If this Material is right to left, reverse the index into the buffers
	if constexpr (REVERSED) {
		index = (BATCH_SIZE - 1) - index;
	}

	float3 ray_direction = material_buffer->direction.get(index);
	RayHit hit           = material_buffer->hits     .get(index);

	int pixel_index = material_buffer->pixel_index[index];
	
	float3 throughput;
	if (bounce == 0) {
		throughput = make_float3(1.0f); // Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		throughput = material_buffer->throughput.get(index);
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

	// Make sure the normal is always pointing outwards
	bool entering_material = dot(ray_direction, normal) < 0.0f;
	if (!entering_material) {
		normal = -normal;
	}

	// Load and propagate Ray Cone
	float cone_angle;
	float cone_width;
	if (config.enable_mipmapping) {
		if (bounce == 0) {
			cone_angle = camera.pixel_spread_angle;
			cone_width = cone_angle * hit.t;
		} else {
			float2 cone = material_buffer->cone[index];
			cone_angle = cone.x;
			cone_width = cone.y + cone_angle * hit.t;
		}
	}

	// Calculate texture level of detail
	float mesh_scale = mesh_get_scale(hit.mesh_id);

	LOD lod;
	if constexpr (BSDF::HAS_ALBEDO) {
		if (config.enable_mipmapping) {
			float3 geometric_normal = cross(hit_triangle.position_edge_1, hit_triangle.position_edge_2);
			float  triangle_area_inv = 1.0f / length(geometric_normal);
			geometric_normal *= triangle_area_inv; // Normalize

			if (bounce == 0) {
				// First bounce uses anisotrpoic LOD
				float3 ellipse_axis_1, ellipse_axis_2;
				ray_cone_get_ellipse_axes(ray_direction, normal, cone_width, ellipse_axis_1, ellipse_axis_2);

				ray_cone_get_texture_gradients(
					mesh_scale,
					geometric_normal,
					triangle_area_inv,
					hit_triangle.position_0,  hit_triangle.position_edge_1,  hit_triangle.position_edge_2,
					hit_triangle.tex_coord_0, hit_triangle.tex_coord_edge_1, hit_triangle.tex_coord_edge_2,
					hit_point_local, tex_coord,
					ellipse_axis_1, ellipse_axis_2,
					lod.aniso.gradient_1, lod.aniso.gradient_2
				);
			} else {
				// Subsequent bounces use isotropic LOD
				float lod_triangle = sqrtf(triangle_get_lod(mesh_scale, triangle_area_inv, hit_triangle.tex_coord_edge_1, hit_triangle.tex_coord_edge_2));
				float lod_ray_cone = ray_cone_get_lod(ray_direction, normal, cone_width);
				lod.iso.lod = log2f(lod_triangle * lod_ray_cone);
			}
		}
	}
	
	// Calulate new Ray Cone angle based on Mesh curvature
	if (config.enable_mipmapping) {
		float curvature = triangle_get_curvature(
			hit_triangle.position_edge_1,
			hit_triangle.position_edge_2,
			hit_triangle.normal_edge_1,
			hit_triangle.normal_edge_2
		) / mesh_scale;
		
		cone_angle -= 2.0f * curvature * fabsf(cone_width) / dot(normal, ray_direction); // Eq. 5 (Akenine-Möller 2021)
	}

	// Construct TBN frame
	float3 tangent, bitangent;
	orthonormal_basis(normal, tangent, bitangent);

	float3 omega_i = world_to_local(-ray_direction, tangent, bitangent, normal);

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
	bsdf.init(bounce, entering_material, material_id, tex_coord, lod);

	bsdf.attenuate(bounce, pixel_index, throughput, hit.t);

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
	if (config.enable_next_event_estimation && lights_total_weight > 0.0f && bsdf.is_mis_eligable()) {
		float2 rand_light    = random<SampleDimension::NEE_LIGHT>   (pixel_index, bounce, sample_index);
		float2 rand_triangle = random<SampleDimension::NEE_TRIANGLE>(pixel_index, bounce, sample_index);

		// Pick random Light
		int light_mesh_id;
		int light_triangle_id = sample_light(rand_light.x, rand_light.y, light_mesh_id);

		// Pick random point on the Light
		float2 light_uv = sample_triangle(rand_triangle.x, rand_triangle.y);

		// Obtain the Light's position and normal
		TrianglePosNor light = triangle_get_positions_and_normals(light_triangle_id);

		float3 light_point;
		float3 light_normal;
		triangle_barycentric(light, light_uv.x, light_uv.y, light_point, light_normal);

		// Transform into world space
		Matrix3x4 light_world = mesh_get_transform(light_mesh_id);
		matrix3x4_transform_position (light_world, light_point);
		matrix3x4_transform_direction(light_world, light_normal);

		light_normal = normalize(light_normal);

		float3 to_light = light_point - hit_point;
		float distance_to_light_squared = dot(to_light, to_light);
		float distance_to_light         = sqrtf(distance_to_light_squared);

		// Normalize the vector to the light
		to_light /= distance_to_light;

		float cos_theta_light = fabsf(dot(to_light, light_normal));
		float cos_theta_hit = dot(to_light, normal);

		int light_material_id = mesh_get_material_id(light_mesh_id);
		MaterialLight material_light = material_as_light(light_material_id);

		float3 bsdf_value;
		float  bsdf_pdf;
		bool valid = bsdf.eval(to_light, cos_theta_hit, bsdf_value, bsdf_pdf);
		
		if (valid) {
			float light_power = luminance(material_light.emission.x, material_light.emission.y, material_light.emission.z);
			float light_pdf   = light_power * distance_to_light_squared / (cos_theta_light * lights_total_weight);

			float mis_weight;
			if (config.enable_multiple_importance_sampling) {
				mis_weight = power_heuristic(light_pdf, bsdf_pdf);
			} else {
				mis_weight = 1.0f;
			}

			float3 illumination = throughput * bsdf_value * material_light.emission * mis_weight / light_pdf;

			// Emit Shadow Ray
			int shadow_ray_index = atomic_agg_inc(&buffer_sizes.shadow[bounce]);

			ray_buffer_shadow.ray_origin   .set(shadow_ray_index, ray_origin_epsilon_offset(hit_point, to_light, normal));
			ray_buffer_shadow.ray_direction.set(shadow_ray_index, to_light);

			ray_buffer_shadow.max_distance[shadow_ray_index] = distance_to_light - 2.0f * EPSILON;

			ray_buffer_shadow.illumination_and_pixel_index[shadow_ray_index] = make_float4(
				illumination.x,
				illumination.y,
				illumination.z,
				__int_as_float(pixel_index)
			);
		}
	}

	// Sample BSDF
	float3 direction_out;
	float pdf;
	bool valid = bsdf.sample(throughput, direction_out, pdf);

	if (!valid) return;

	float3 origin_out = ray_origin_epsilon_offset(hit_point, direction_out, normal);

	// Emit next Ray
	int index_out = atomic_agg_inc(&buffer_sizes.trace[bounce + 1]);

	ray_buffer_trace.origin   .set(index_out, origin_out);
	ray_buffer_trace.direction.set(index_out, direction_out);

	if (config.enable_mipmapping) {
		ray_buffer_trace.cone[index_out] = make_float2(cone_angle, cone_width);
	}

	ray_buffer_trace.pixel_index_and_mis_eligable[index_out] = pixel_index | (bsdf.is_mis_eligable() << 31);
	ray_buffer_trace.throughput.set(index_out, throughput);

	ray_buffer_trace.last_pdf[index_out] = pdf;	
}

extern "C" __global__ void kernel_shade_diffuse(int bounce, int sample_index) {
	shade_material<BSDFDiffuse, &ray_buffer_shade_diffuse_and_plastic, false>(bounce, sample_index, buffer_sizes.diffuse[bounce]);
}

extern "C" __global__ void kernel_shade_plastic(int bounce, int sample_index) {
	shade_material<BSDFPlastic, &ray_buffer_shade_diffuse_and_plastic, true>(bounce, sample_index, buffer_sizes.plastic[bounce]);
}

extern "C" __global__ void kernel_shade_dielectric(int bounce, int sample_index) {
	shade_material<BSDFDielectric, &ray_buffer_shade_dielectric_and_conductor, false>(bounce, sample_index, buffer_sizes.dielectric[bounce]);
}

extern "C" __global__ void kernel_shade_conductor(int bounce, int sample_index) {
	shade_material<BSDFConductor, &ray_buffer_shade_dielectric_and_conductor, true>(bounce, sample_index, buffer_sizes.conductor[bounce]);
}

extern "C" __global__ void kernel_accumulate(float frames_accumulated) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float4 direct   = frame_buffer_direct  [pixel_index];
	float4 indirect = frame_buffer_indirect[pixel_index];

	float4 colour = direct + indirect;

	if (config.enable_albedo) {
		colour *= frame_buffer_albedo[pixel_index];
	}

	if (frames_accumulated > 0.0f) {
		float4 colour_prev = accumulator.get(x, y);

		colour = colour_prev + (colour - colour_prev) / frames_accumulated; // Online average
	}

//	if (isnan(colour.x + colour.y + colour.z)) colour = make_float4(1,0,1,1);

	accumulator.set(x, y, colour);
}
