#include "cudart/vector_types.h"
#include "cudart/cuda_math.h"

#include "Common.h"

#define INFINITY ((float)(1e+300 * 1e+300))

__device__ __constant__ int screen_width;
__device__ __constant__ int screen_pitch;
__device__ __constant__ int screen_height;

__device__ __constant__ Config config;

#include "Util.h"
#include "Material.h"
#include "Sky.h"
#include "RayCone.h"

// Frame Buffers
__device__ __constant__ float4 * frame_buffer_albedo;
__device__ __constant__ float4 * frame_buffer_direct;
__device__ __constant__ float4 * frame_buffer_indirect;

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

	unsigned ray_pixel_index_and_mis_eligable = ray_buffer_trace.pixel_index_and_mis_eligable[index];
	int      ray_pixel_index = ray_pixel_index_and_mis_eligable & ~(0b11 << 31);

	int x = ray_pixel_index % screen_pitch;
	int y = ray_pixel_index / screen_pitch;

	bool mis_eligable = ray_pixel_index_and_mis_eligable >> 31;

	float3 ray_throughput;
	if (bounce == 0) {
		ray_throughput = make_float3(1.0f); // Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		ray_throughput = ray_buffer_trace.throughput.get(index);
	}

	// If we didn't hit anything, sample the Sky
	if (hit.triangle_id == INVALID) {
		float3 illumination = ray_throughput * sample_sky(ray_direction);

		if (bounce == 0) {
			if (config.enable_albedo || config.enable_svgf) {
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

	// Get the Material of the Mesh we hit
	int material_id = mesh_get_material_id(hit.mesh_id);
	MaterialType material_type = material_get_type(material_id);

	if (bounce == 0 && pixel_query.pixel_index == ray_pixel_index) {
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
			float3 illumination = ray_throughput * material_light.emission;

			if (bounce == 0) {
				if (config.enable_albedo || config.enable_svgf) {
					frame_buffer_albedo[ray_pixel_index] = make_float4(1.0f);
				}
				frame_buffer_direct[ray_pixel_index] = make_float4(material_light.emission);
			} else if (bounce == 1) {
				frame_buffer_direct[ray_pixel_index] += make_float4(illumination);
			} else {
				frame_buffer_indirect[ray_pixel_index] += make_float4(illumination);
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
			float3 illumination = ray_throughput * material_light.emission * mis_weight;

			assert(bounce != 0);
			if (bounce == 1) {
				frame_buffer_direct[ray_pixel_index] += make_float4(illumination);
			} else {
				frame_buffer_indirect[ray_pixel_index] += make_float4(illumination);
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
		float3 throughput_with_albedo = ray_throughput * make_float3(frame_buffer_albedo[ray_pixel_index]);

		float survival_probability  = saturate(vmax_max(throughput_with_albedo.x, throughput_with_albedo.y, throughput_with_albedo.z));
		float rand_russian_roulette = random<SampleDimension::RUSSIAN_ROULETTE>(ray_pixel_index, bounce, sample_index).x;

		if (rand_russian_roulette > survival_probability) {
			return;
		}

		ray_throughput /= survival_probability;
	}

	switch (material_type) {
		case MaterialType::DIFFUSE: {
			int index_out = atomic_agg_inc(&buffer_sizes.diffuse[bounce]);

			ray_buffer_shade_diffuse_and_plastic.direction.set(index_out, ray_direction);

			if (bounce > 0 && config.enable_mipmapping) ray_buffer_shade_diffuse_and_plastic.cone[index_out] = ray_buffer_trace.cone[index];

			ray_buffer_shade_diffuse_and_plastic.hits.set(index_out, hit);

			ray_buffer_shade_diffuse_and_plastic.pixel_index[index_out] = ray_pixel_index;
			if (bounce > 0) ray_buffer_shade_diffuse_and_plastic.throughput.set(index_out, ray_throughput);

			break;
		}

		case MaterialType::PLASTIC: {
			// Plastic Material buffer is shared with Diffuse Material buffer but grows in the opposite direction
			int index_out = (BATCH_SIZE - 1) - atomic_agg_inc(&buffer_sizes.plastic[bounce]);

			ray_buffer_shade_diffuse_and_plastic.direction.set(index_out, ray_direction);

			if (bounce > 0 && config.enable_mipmapping) ray_buffer_shade_diffuse_and_plastic.cone[index_out] = ray_buffer_trace.cone[index];

			ray_buffer_shade_diffuse_and_plastic.hits.set(index_out, hit);

			ray_buffer_shade_diffuse_and_plastic.pixel_index[index_out] = ray_pixel_index;
			if (bounce > 0) ray_buffer_shade_diffuse_and_plastic.throughput.set(index_out, ray_throughput);

			break;
		}

		case MaterialType::DIELECTRIC: {
			int index_out = atomic_agg_inc(&buffer_sizes.dielectric[bounce]);

			ray_buffer_shade_dielectric_and_conductor.direction.set(index_out, ray_direction);

			if (bounce > 0 && config.enable_mipmapping) ray_buffer_shade_dielectric_and_conductor.cone[index_out] = ray_buffer_trace.cone[index];

			ray_buffer_shade_dielectric_and_conductor.hits.set(index_out, hit);

			ray_buffer_shade_dielectric_and_conductor.pixel_index[index_out] = ray_pixel_index;
			if (bounce > 0) ray_buffer_shade_dielectric_and_conductor.throughput.set(index_out, ray_throughput);

			break;
		}

		case MaterialType::CONDUCTOR: {
			// Conductor Material buffer is shared with Dielectric Material buffer but grows in the opposite direction
			int index_out = (BATCH_SIZE - 1) - atomic_agg_inc(&buffer_sizes.conductor[bounce]);

			ray_buffer_shade_dielectric_and_conductor.direction.set(index_out, ray_direction);

			if (bounce > 0 && config.enable_mipmapping) ray_buffer_shade_dielectric_and_conductor.cone[index_out] = ray_buffer_trace.cone[index];

			ray_buffer_shade_dielectric_and_conductor.hits.set(index_out, hit);

			ray_buffer_shade_dielectric_and_conductor.pixel_index[index_out] = ray_pixel_index;
			if (bounce > 0) ray_buffer_shade_dielectric_and_conductor.throughput.set(index_out, ray_throughput);

			break;
		}
	}
}

__device__ inline float3 sample_albedo(
	int                       bounce,
	const float3            & material_diffuse,
	int                       material_texture_id,
	const RayHit            & hit,
	const TrianglePosNorTex & hit_triangle,
	const float3            & hit_point_local,
	const float3            & hit_normal,
	const float2            & hit_tex_coord,
	const float3            & ray_direction,
	const float2            * cone_buffer,
	int                       cone_buffer_index,
	float                   & cone_angle,
	float                   & cone_width
) {
	float3 albedo;

	float3 geometric_normal = cross(hit_triangle.position_edge_1, hit_triangle.position_edge_2);
	float  triangle_area_inv = 1.0f / length(geometric_normal);
	geometric_normal *= triangle_area_inv; // Normalize

	float mesh_scale = mesh_get_scale(hit.mesh_id);

	if (bounce == 0) {
		cone_angle = camera.pixel_spread_angle;
		cone_width = cone_angle * hit.t;

		float3 ellipse_axis_1, ellipse_axis_2; ray_cone_get_ellipse_axes(ray_direction, hit_normal, cone_width, ellipse_axis_1, ellipse_axis_2);

		float2 gradient_1, gradient_2; ray_cone_get_texture_gradients(
			mesh_scale,
			geometric_normal,
			triangle_area_inv,
			hit_triangle.position_0,  hit_triangle.position_edge_1,  hit_triangle.position_edge_2,
			hit_triangle.tex_coord_0, hit_triangle.tex_coord_edge_1, hit_triangle.tex_coord_edge_2,
			hit_point_local, hit_tex_coord,
			ellipse_axis_1, ellipse_axis_2,
			gradient_1, gradient_2
		);

		// Anisotropic sampling
		albedo = material_get_albedo(material_diffuse, material_texture_id, hit_tex_coord.x, hit_tex_coord.y, gradient_1, gradient_2);
	} else {
		float2 cone = cone_buffer[cone_buffer_index];
		cone_angle = cone.x;
		cone_width = cone.y + cone_angle * hit.t;

		float2 tex_size = texture_get_size(material_texture_id);

		float lod_triangle = sqrtf(tex_size.x * tex_size.y * triangle_get_lod(mesh_scale, triangle_area_inv, hit_triangle.tex_coord_edge_1, hit_triangle.tex_coord_edge_2));
		float lod_ray_cone = ray_cone_get_lod(ray_direction, hit_normal, cone_width);
		float lod = log2f(lod_triangle * lod_ray_cone);

		// Trilinear sampling
		albedo = material_get_albedo(material_diffuse, material_texture_id, hit_tex_coord.x, hit_tex_coord.y, lod);
	}

	float curvature = triangle_get_curvature(
		hit_triangle.position_edge_1,
		hit_triangle.position_edge_2,
		hit_triangle.normal_edge_1,
		hit_triangle.normal_edge_2
	) / mesh_scale;

	cone_angle += -2.0f * curvature * fabsf(cone_width / dot(hit_normal, ray_direction)); // Eq. 5 (Akenine-Möller 2021)

	return albedo;
}

template<typename BRDFEvaluator>
__device__ inline void nee_sample(
	int pixel_index,
	int bounce,
	int sample_index,
	const float3 & hit_point,
	const float3 & hit_normal,
	const float3 & throughput,
	BRDFEvaluator brdf_evaluator
) {
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
	float cos_theta_hit = dot(to_light, hit_normal);

	int light_material_id = mesh_get_material_id(light_mesh_id);
	MaterialLight material_light = material_as_light(light_material_id);

	float3 brdf;
	float  brdf_pdf;
	bool valid = brdf_evaluator(to_light, cos_theta_hit, brdf, brdf_pdf);

	if (!valid) return;
	assert(brdf_pdf != 0.0f);

	float light_power = luminance(material_light.emission.x, material_light.emission.y, material_light.emission.z);
	float light_pdf   = light_power * distance_to_light_squared / (cos_theta_light * lights_total_weight);

	float mis_weight;
	if (config.enable_multiple_importance_sampling) {
		mis_weight = power_heuristic(light_pdf, brdf_pdf);
	} else {
		mis_weight = 1.0f;
	}

	float3 illumination = throughput * brdf * material_light.emission * mis_weight / light_pdf;

	int shadow_ray_index = atomic_agg_inc(&buffer_sizes.shadow[bounce]);

	ray_buffer_shadow.ray_origin   .set(shadow_ray_index, ray_origin_epsilon_offset(hit_point, to_light, hit_normal));
	ray_buffer_shadow.ray_direction.set(shadow_ray_index, to_light);

	ray_buffer_shadow.max_distance[shadow_ray_index] = distance_to_light - 2.0f * EPSILON;

	ray_buffer_shadow.illumination_and_pixel_index[shadow_ray_index] = make_float4(
		illumination.x,
		illumination.y,
		illumination.z,
		__int_as_float(pixel_index)
	);
}

extern "C" __global__ void kernel_shade_diffuse(int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.diffuse[bounce]) return;

	float3 ray_direction = ray_buffer_shade_diffuse_and_plastic.direction.get(index);
	RayHit hit           = ray_buffer_shade_diffuse_and_plastic.hits     .get(index);

	int ray_pixel_index = ray_buffer_shade_diffuse_and_plastic.pixel_index[index];
	int x = ray_pixel_index % screen_pitch;
	int y = ray_pixel_index / screen_pitch;

	float3 ray_throughput;
	if (bounce == 0) {
		ray_throughput = make_float3(1.0f); // Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		ray_throughput = ray_buffer_shade_diffuse_and_plastic.throughput.get(index);
	}

	int material_id = mesh_get_material_id(hit.mesh_id);
	MaterialDiffuse material = material_as_diffuse(material_id);

	// Obtain hit Triangle position, normal, and texture coordinates
	TrianglePosNorTex hit_triangle = triangle_get_positions_normals_and_tex_coords(hit.triangle_id);

	float3 hit_point;
	float3 hit_normal;
	float2 hit_tex_coord;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, hit_normal, hit_tex_coord);

	float3 hit_point_local = hit_point; // Keep copy of the untransformed hit point in local space

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, hit_normal);

	hit_normal = normalize(hit_normal);
	if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	// Sample albedo
	float cone_angle;
	float cone_width;
	float3 albedo;
	if (config.enable_mipmapping) {
		albedo = sample_albedo(
			bounce,
			material.diffuse,
			material.texture_id,
			hit,
			hit_triangle,
			hit_point_local,
			hit_normal,
			hit_tex_coord,
			ray_direction,
			ray_buffer_shade_diffuse_and_plastic.cone,
			index,
			cone_angle, cone_width
		);
	} else {
		albedo = material_get_albedo(material.diffuse, material.texture_id, hit_tex_coord.x, hit_tex_coord.y);
	}

	if (bounce > 0) {
		ray_throughput *= albedo;
	} else if (config.enable_albedo || config.enable_svgf) {
		frame_buffer_albedo[ray_pixel_index] = make_float4(albedo);
	}

	if (bounce == 0 && config.enable_svgf) {
		float3 hit_point_prev = hit_point_local;

		Matrix3x4 world_prev = mesh_get_transform_prev(hit.mesh_id);
		matrix3x4_transform_position(world_prev, hit_point_prev);

		svgf_set_gbuffers(x, y, hit, hit_point, hit_normal, hit_point_prev);
	}

	if (config.enable_next_event_estimation && lights_total_weight > 0.0f) {
		nee_sample(ray_pixel_index, bounce, sample_index, hit_point, hit_normal, ray_throughput, [&](const float3 & to_light, float cos_theta, float3 & brdf, float & pdf) {
			if (cos_theta <= 0.0f) return false;

			brdf = make_float3(cos_theta * ONE_OVER_PI);
			pdf  = cos_theta * ONE_OVER_PI;

			return pdf_is_valid(pdf);
		});
	}

	float3 hit_tangent, hit_binormal; orthonormal_basis(hit_normal, hit_tangent, hit_binormal);

	float2 rand_brdf = random<SampleDimension::BRDF>(ray_pixel_index, bounce, sample_index);
	float3 direction_local = sample_cosine_weighted_direction(rand_brdf.x, rand_brdf.y);

	float3 direction_out = local_to_world(direction_local, hit_tangent, hit_binormal, hit_normal);
	float3 origin_out    = ray_origin_epsilon_offset(hit_point, direction_out, hit_normal);

	int index_out = atomic_agg_inc(&buffer_sizes.trace[bounce + 1]);

	ray_buffer_trace.origin   .set(index_out, origin_out);
	ray_buffer_trace.direction.set(index_out, direction_out);

	if (config.enable_mipmapping) {
		ray_buffer_trace.cone[index_out] = make_float2(cone_angle, cone_width);
	}

	ray_buffer_trace.pixel_index_and_mis_eligable[index_out] = ray_pixel_index | (true << 31);
	ray_buffer_trace.throughput.set(index_out, ray_throughput);

	ray_buffer_trace.last_pdf[index_out] = fabsf(dot(direction_out, hit_normal)) * ONE_OVER_PI;
}

extern "C" __global__ void kernel_shade_plastic(int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.plastic[bounce]) return;

	index = (BATCH_SIZE - 1) - index;

	float3 ray_direction = ray_buffer_shade_diffuse_and_plastic.direction.get(index);
	RayHit hit           = ray_buffer_shade_diffuse_and_plastic.hits     .get(index);

	int ray_pixel_index = ray_buffer_shade_diffuse_and_plastic.pixel_index[index];
	int x = ray_pixel_index % screen_pitch;
	int y = ray_pixel_index / screen_pitch;

	float3 ray_throughput;
	if (bounce == 0) {
		ray_throughput = make_float3(1.0f); // Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		ray_throughput = ray_buffer_shade_diffuse_and_plastic.throughput.get(index);
	}

	int material_id = mesh_get_material_id(hit.mesh_id);
	MaterialPlastic material = material_as_plastic(material_id);

	// Obtain hit Triangle position, normal, and texture coordinates
	TrianglePosNorTex hit_triangle = triangle_get_positions_normals_and_tex_coords(hit.triangle_id);

	float3 hit_point;
	float3 hit_normal;
	float2 hit_tex_coord;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, hit_normal, hit_tex_coord);

	float3 hit_point_local = hit_point; // Keep copy of the untransformed hit point in local space

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, hit_normal);

	hit_normal = normalize(hit_normal);
	if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	// Sample albedo
	float cone_angle;
	float cone_width;
	float3 albedo;
	if (config.enable_mipmapping) {
		albedo = sample_albedo(
			bounce,
			material.diffuse,
			material.texture_id,
			hit,
			hit_triangle,
			hit_point_local,
			hit_normal,
			hit_tex_coord,
			ray_direction,
			ray_buffer_shade_diffuse_and_plastic.cone,
			index,
			cone_angle, cone_width
		);
	} else {
		albedo = material_get_albedo(material.diffuse, material.texture_id, hit_tex_coord.x, hit_tex_coord.y);
	}

	if (bounce == 0 && (config.enable_albedo || config.enable_svgf)) {
		frame_buffer_albedo[ray_pixel_index] = make_float4(1.0f);
	}

	if (bounce == 0 && config.enable_svgf) {
		float3 hit_point_prev = hit_point_local;

		Matrix3x4 world_prev = mesh_get_transform_prev(hit.mesh_id);
		matrix3x4_transform_position(world_prev, hit_point_prev);

		svgf_set_gbuffers(x, y, hit, hit_point, hit_normal, hit_point_prev);
	}

	float3 hit_tangent, hit_binormal; orthonormal_basis(hit_normal, hit_tangent, hit_binormal);

	float3 omega_i = world_to_local(-ray_direction, hit_tangent, hit_binormal, hit_normal);

	float alpha_x = material.roughness;
	float alpha_y = material.roughness;
	
	constexpr float ETA = 1.0f / 1.5f;
	constexpr float TIR_COMPENSATION = 0.596345782f; // Hemispherical integral of fresnel * cos(theta)

	if (config.enable_next_event_estimation && lights_total_weight > 0.0f) {
		nee_sample(ray_pixel_index, bounce, sample_index, hit_point, hit_normal, ray_throughput, [&](const float3 & to_light, float cos_theta, float3 & brdf, float & pdf) {
			if (cos_theta <= 0.0f) return false;

			float3 omega_o = world_to_local(to_light, hit_tangent, hit_binormal, hit_normal);
			float3 omega_m = normalize(omega_i + omega_o);

			// Specular component
			float F  = fresnel_dielectric(dot(omega_i, omega_m), ETA);
			float D  = ggx_D (omega_m, alpha_x, alpha_y);
			float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
			float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

			float3 brdf_specular = make_float3(F * G2 * D / (4.0f * omega_i.z));

			// Diffuse component
			float F_i = fresnel_dielectric(omega_i.z, ETA);
			float F_o = fresnel_dielectric(omega_o.z, ETA);

			float3 brdf_diffuse = ETA*ETA * (1.0f - F_i) * (1.0f - F_o) * albedo * ONE_OVER_PI / (1.0f - albedo * TIR_COMPENSATION) * omega_o.z;

			float pdf_specular = G1 * D / (4.0f * omega_i.z);
			float pdf_diffuse  = omega_o.z * ONE_OVER_PI;

			pdf  = lerp(pdf_diffuse, pdf_specular, F_i);
			brdf = brdf_specular + brdf_diffuse; // BRDF * cos(theta_o)

			return pdf_is_valid(pdf);
		});
	}

	float  rand_fresnel = random<SampleDimension::RUSSIAN_ROULETTE>(ray_pixel_index, bounce, sample_index).y;
	float2 rand_brdf    = random<SampleDimension::BRDF>            (ray_pixel_index, bounce, sample_index);

	float F_i = fresnel_dielectric(omega_i.z, ETA);

	float3 omega_m;
	float3 omega_o;
	if (rand_fresnel < F_i) {
		// Sample specular component
		omega_m = sample_visible_normals_ggx(omega_i, material.roughness, material.roughness, rand_brdf.x, rand_brdf.y);
		omega_o = reflect(-omega_i, omega_m);
	} else {
		// Sample diffuse component
		omega_o = sample_cosine_weighted_direction(rand_brdf.x, rand_brdf.y);
		omega_m = normalize(omega_i + omega_o);
	}

	if (omega_m.z < 0.0f) return; // Wrong hemisphere

	// Specular component
	float F  = fresnel_dielectric(dot(omega_i, omega_m), ETA);
	float D  = ggx_D (omega_m, alpha_x, alpha_y);
	float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
	float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

	float3 brdf_specular = make_float3(F * G2 * D / (4.0f * omega_i.z));

	// Diffuse component
	float F_o = fresnel_dielectric(omega_o.z, ETA);

	float3 brdf_diffuse = ETA*ETA * (1.0f - F_i) * (1.0f - F_o) * albedo * ONE_OVER_PI / (1.0f - albedo * TIR_COMPENSATION) * omega_o.z;

	// PDFs
	float pdf_specular = G1 * D / (4.0f * omega_i.z);
	float pdf_diffuse  = omega_o.z * ONE_OVER_PI;
	float pdf          = lerp(pdf_diffuse, pdf_specular, F_i);

	if (!pdf_is_valid(pdf)) return;

	ray_throughput *= (brdf_specular + brdf_diffuse) / pdf; // BRDF * cos(theta) / pdf

	float3 direction_out = local_to_world(omega_o, hit_tangent, hit_binormal, hit_normal);
	float3 origin_out = ray_origin_epsilon_offset(hit_point, direction_out, hit_normal);

	int index_out = atomic_agg_inc(&buffer_sizes.trace[bounce + 1]);

	ray_buffer_trace.origin   .set(index_out, origin_out);
	ray_buffer_trace.direction.set(index_out, direction_out);

	if (config.enable_mipmapping) {
		ray_buffer_trace.cone[index_out] = make_float2(cone_angle, cone_width);
	}

	ray_buffer_trace.pixel_index_and_mis_eligable[index_out] = ray_pixel_index | (true << 31);
	ray_buffer_trace.throughput.set(index_out, ray_throughput);

	ray_buffer_trace.last_pdf[index_out] = pdf;
}

extern "C" __global__ void kernel_shade_dielectric(int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.dielectric[bounce]) return;

	float3 ray_direction = ray_buffer_shade_dielectric_and_conductor.direction.get(index);
	RayHit hit           = ray_buffer_shade_dielectric_and_conductor.hits     .get(index);

	int ray_pixel_index = ray_buffer_shade_dielectric_and_conductor.pixel_index[index];

	float3 ray_throughput;
	if (bounce == 0) {
		ray_throughput = make_float3(1.0f); // Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		ray_throughput = ray_buffer_shade_dielectric_and_conductor.throughput.get(index);
	}

	ASSERT(hit.triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	int material_id = mesh_get_material_id(hit.mesh_id);
	MaterialDielectric material = material_as_dielectric(material_id);

	// Obtain hit Triangle position, normal, and texture coordinates
	TrianglePosNor hit_triangle = triangle_get_positions_and_normals(hit.triangle_id);

	float3 hit_point;
	float3 hit_normal;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, hit_normal);

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, hit_normal);

	hit_normal = normalize(hit_normal);

	bool entering_material = dot(ray_direction, hit_normal) < 0.0f;
	if (!entering_material) {
		hit_normal = -hit_normal;

		// Lambert-Beer Law
		// NOTE: does not take into account e.g. nested dielectrics or diffuse inside dielectric!
		ray_throughput.x *= expf(material.negative_absorption.x * hit.t);
		ray_throughput.y *= expf(material.negative_absorption.y * hit.t);
		ray_throughput.z *= expf(material.negative_absorption.z * hit.t);
	}

	// Construct orthonormal basis
	float3 hit_tangent, hit_binormal;
	orthonormal_basis(hit_normal, hit_tangent, hit_binormal);

	float3 omega_i = world_to_local(-ray_direction, hit_tangent, hit_binormal, hit_normal);

	float eta = entering_material ? 1.0f / material.index_of_refraction : material.index_of_refraction;

	float alpha_x = material.roughness;
	float alpha_y = material.roughness;
	
	if (config.enable_next_event_estimation && lights_total_weight > 0.0f && material.roughness >= ROUGHNESS_CUTOFF) {
		nee_sample(ray_pixel_index, bounce, sample_index, hit_point, hit_normal, ray_throughput, [&](const float3 & to_light, float cos_theta, float3 & brdf, float & pdf) {
			float3 omega_o = world_to_local(to_light, hit_tangent, hit_binormal, hit_normal);

			bool reflected = omega_o.z >= 0.0f; // Same sign means reflection, alternate signs means transmission

			float3 omega_m;
			if (reflected) {
				omega_m = normalize(omega_i + omega_o);
			} else {
				omega_m = normalize(eta * omega_i + omega_o);
			}
			omega_m *= sign(omega_m.z);

			float i_dot_m = abs_dot(omega_i, omega_m);
			float o_dot_m = abs_dot(omega_o, omega_m);

			float F  = fresnel_dielectric(i_dot_m, eta);
			float D  = ggx_D (omega_m, alpha_x, alpha_y);
			float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
			float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

			if (reflected) {
				pdf = F * G1 * D / (4.0f * omega_i.z);

				brdf = make_float3(F * G2 * D / (4.0f * omega_i.z)); // BRDF times cos(theta_o)
			} else {
				if (F >= 0.999f) return false; // TIR, no transmission possible

				pdf = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));

				brdf = eta * eta * make_float3((1.0f - F) * G2 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m))); // BRDF times cos(theta_o)
			}

			return pdf_is_valid(pdf);
		});
	}

	float  rand_fresnel = random<SampleDimension::RUSSIAN_ROULETTE>(ray_pixel_index, bounce, sample_index).y;
	float2 rand_brdf    = random<SampleDimension::BRDF>            (ray_pixel_index, bounce, sample_index);

	float3 omega_m = sample_visible_normals_ggx(omega_i, material.roughness, material.roughness, rand_brdf.x, rand_brdf.y);

	float F = fresnel_dielectric(abs_dot(omega_i, omega_m), eta);
	bool reflected = rand_fresnel < F;

	float3 omega_o;
	if (reflected) {
		omega_o = 2.0f * dot(omega_i, omega_m) * omega_m - omega_i;
	} else {
		float k = 1.0f - eta*eta * (1.0f - square(dot(omega_i, omega_m)));
		omega_o = (eta * abs_dot(omega_i, omega_m) - sqrtf(k)) * omega_m - eta * omega_i;
	}

	if (reflected ^ (omega_o.z >= 0.0f)) return; // Hemisphere check: reflection should have positive z, transmission negative z

	float D  = ggx_D (omega_m, alpha_x, alpha_y);
	float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
	float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

	float i_dot_m = abs_dot(omega_i, omega_m);
	float o_dot_m = abs_dot(omega_o, omega_m);

	float pdf;
	if (reflected) {
		pdf = F * G1 * D / (4.0f * omega_i.z);
	} else {
		pdf = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));

		ray_throughput *= eta*eta; // Account for solid angle compression
	}

	ray_throughput *= G2 / G1; // BRDF * cos(theta_o) / pdf (same for reflection and transmission)

	float3 direction_out = local_to_world(omega_o, hit_tangent, hit_binormal, hit_normal);
	float3 origin_out    = ray_origin_epsilon_offset(hit_point, direction_out, hit_normal);

	if (bounce == 0 && (config.enable_albedo || config.enable_svgf)) {
		frame_buffer_albedo[ray_pixel_index] = make_float4(1.0f);
	}

	int index_out = atomic_agg_inc(&buffer_sizes.trace[bounce + 1]);

	ray_buffer_trace.origin   .set(index_out, origin_out);
	ray_buffer_trace.direction.set(index_out, direction_out);

	if (config.enable_mipmapping) {
		float cone_angle;
		float cone_width;
		if (bounce == 0) {
			cone_angle = camera.pixel_spread_angle;
			cone_width = 0.0f;
		} else {
			float2 cone = ray_buffer_shade_dielectric_and_conductor.cone[index];
			cone_angle = cone.x;
			cone_width = cone.y;
		}

		float mesh_scale = mesh_get_scale(hit.mesh_id);

		float curvature = triangle_get_curvature(
			hit_triangle.position_edge_1,
			hit_triangle.position_edge_2,
			hit_triangle.normal_edge_1,
			hit_triangle.normal_edge_2
		) / mesh_scale;

		cone_width += cone_angle * hit.t;
		cone_angle += -2.0f * curvature * fabsf(cone_width) / dot(hit_normal, ray_direction); // Eq. 5 (Akenine-Möller 2021)

		ray_buffer_trace.cone[index_out] = make_float2(cone_angle, cone_width);
	}

	ray_buffer_trace.pixel_index_and_mis_eligable[index_out] = ray_pixel_index | ((material.roughness >= ROUGHNESS_CUTOFF) << 31);
	ray_buffer_trace.throughput.set(index_out, ray_throughput);

	ray_buffer_trace.last_pdf[index_out] = pdf;
}

extern "C" __global__ void kernel_shade_conductor(int bounce, int sample_index) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.conductor[bounce]) return;

	index = (BATCH_SIZE - 1) - index;

	float3 ray_direction = ray_buffer_shade_dielectric_and_conductor.direction.get(index);

	RayHit hit = ray_buffer_shade_dielectric_and_conductor.hits.get(index);

	int ray_pixel_index = ray_buffer_shade_dielectric_and_conductor.pixel_index[index];
	int x = ray_pixel_index % screen_pitch;
	int y = ray_pixel_index / screen_pitch;

	float3 ray_throughput;
	if (bounce == 0) {
		ray_throughput = make_float3(1.0f);	// Throughput is known to be (1,1,1) still, skip the global memory load
	} else {
		ray_throughput = ray_buffer_shade_dielectric_and_conductor.throughput.get(index);
	}

	ASSERT(hit.triangle_id != -1, "Ray must have hit something for this Kernel to be invoked!");

	int material_id = mesh_get_material_id(hit.mesh_id);
	MaterialConductor material = material_as_conductor(material_id);

	// Obtain hit Triangle position, normal, and texture coordinates
	TrianglePosNorTex hit_triangle = triangle_get_positions_normals_and_tex_coords(hit.triangle_id);

	float3 hit_point;
	float3 hit_normal;
	float2 hit_tex_coord;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, hit_normal, hit_tex_coord);

	float3 hit_point_local = hit_point; // Keep copy of the untransformed hit point in local space

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, hit_normal);

	hit_normal = normalize(hit_normal);
	if (dot(ray_direction, hit_normal) > 0.0f) hit_normal = -hit_normal;

	// Sample albedo
	float cone_angle;
	float cone_width;
	float3 albedo;
	if (config.enable_mipmapping) {
		albedo = sample_albedo(
			bounce,
			material.diffuse,
			material.texture_id,
			hit,
			hit_triangle,
			hit_point_local,
			hit_normal,
			hit_tex_coord,
			ray_direction,
			ray_buffer_shade_dielectric_and_conductor.cone,
			index,
			cone_angle, cone_width
		);
	} else {
		albedo = material_get_albedo(material.diffuse, material.texture_id, hit_tex_coord.x, hit_tex_coord.y);
	}

	if (bounce > 0) {
		ray_throughput *= albedo;
	} else if (config.enable_albedo || config.enable_svgf) {
		frame_buffer_albedo[ray_pixel_index] = make_float4(albedo);
	}

	if (bounce == 0 && config.enable_svgf) {
		float3 hit_point_prev = hit_point_local;

		Matrix3x4 world_prev = mesh_get_transform_prev(hit.mesh_id);
		matrix3x4_transform_position(world_prev, hit_point_prev);

		svgf_set_gbuffers(x, y, hit, hit_point, hit_normal, hit_point_prev);
	}

	// Construct orthonormal basis
	float3 hit_tangent, hit_binormal;
	orthonormal_basis(hit_normal, hit_tangent, hit_binormal);

	float3 omega_i = world_to_local(-ray_direction, hit_tangent, hit_binormal, hit_normal);
	
	float alpha_x = material.roughness;
	float alpha_y = material.roughness; // TODO: anisotropic
	
	if (config.enable_next_event_estimation && lights_total_weight > 0.0f && material.roughness >= ROUGHNESS_CUTOFF) {
		nee_sample(ray_pixel_index, bounce, sample_index, hit_point, hit_normal, ray_throughput, [&](const float3 & to_light, float cos_theta, float3 & brdf, float & pdf) {
			if (cos_theta <= 0.0f) return false;

			float3 omega_o = world_to_local(to_light, hit_tangent, hit_binormal, hit_normal);
			float3 omega_m = normalize(omega_o + omega_i);
			
			float mu = dot(omega_o, omega_m);
			if (mu <= 0.0f) return false;

			float3 F  = fresnel_conductor(mu, material.eta, material.k);
			float  D  = ggx_D (omega_m, alpha_x, alpha_y);
			float  G1 = ggx_G1(omega_i, alpha_x, alpha_y);
			float  G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

			pdf  =     G1 * D / (4.0f * omega_i.z);
			brdf = F * G2 * D / (4.0f * omega_i.z); // BRDF * cos(theta_o)

			return pdf_is_valid(pdf);
		});
	}

	// Importance sample distribution of normals
	float2 rand_brdf = random<SampleDimension::BRDF>(ray_pixel_index, bounce, sample_index);

	float3 omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_brdf.x, rand_brdf.y);
	float3 omega_o = reflect(-omega_i, omega_m);

	float o_dot_m = dot(omega_o, omega_m);
	if (o_dot_m <= 0.0f) return;

	float3 F  = fresnel_conductor(o_dot_m, material.eta, material.k);
	float  D  = ggx_D (omega_m, alpha_x, alpha_y);
	float  G1 = ggx_G1(omega_i, alpha_x, alpha_y);
	float  G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

	float pdf = G1 * D / (4.0f * omega_i.z);

	ray_throughput *= F * G2 / G1; // BRDF * cos(theta_o) / pdf

	float3 direction_out = local_to_world(omega_o, hit_tangent, hit_binormal, hit_normal);
	float3 origin_out    = ray_origin_epsilon_offset(hit_point, direction_out, hit_normal);

	int index_out = atomic_agg_inc(&buffer_sizes.trace[bounce + 1]);

	ray_buffer_trace.origin   .set(index_out, origin_out);
	ray_buffer_trace.direction.set(index_out, direction_out);

	if (config.enable_mipmapping) {
		ray_buffer_trace.cone[index_out] = make_float2(cone_angle, cone_width);
	}

	ray_buffer_trace.pixel_index_and_mis_eligable[index_out] = ray_pixel_index | ((material.roughness >= ROUGHNESS_CUTOFF) << 31);
	ray_buffer_trace.throughput.set(index_out, ray_throughput);

	ray_buffer_trace.last_pdf[index_out] = pdf;
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
