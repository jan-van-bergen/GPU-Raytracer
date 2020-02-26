#include <vector_types.h>
#include <corecrt_math.h>

#include "cuda_math.h"

#include "../Common.h"

#include "Tracing.h"
#include "Lighting.h"
#include "Sky.h"
#include "Random.h"
#include "Util.h"

surface<void, 2> frame_buffer;

__device__ float3 sample(unsigned & seed, Ray & ray) {
	float3 colour     = make_float3(0.0f);
	float3 throughput = make_float3(1.0f);
	
	bool last_specular = true;

	for (int bounce = 0; bounce < NUM_BOUNCES; bounce++) {
		// Check ray against all triangles
		RayHit hit;
		mbvh_trace(ray, hit);

		// Check if we didn't hit anything
		if (hit.t == INFINITY) {
			return colour + throughput * sample_sky(ray.direction);
		}

		int hit_triangle_id = hit.triangle_id;

		const Material & material = materials[triangles_material_id[hit_triangle_id]];

		float3 hit_point     = barycentric(hit.u, hit.v, triangles_position0 [hit_triangle_id], triangles_position_edge1 [hit_triangle_id], triangles_position_edge2 [hit_triangle_id]);
		float3 hit_normal    = barycentric(hit.u, hit.v, triangles_normal0   [hit_triangle_id], triangles_normal_edge1   [hit_triangle_id], triangles_normal_edge2   [hit_triangle_id]);
		float2 hit_tex_coord = barycentric(hit.u, hit.v, triangles_tex_coord0[hit_triangle_id], triangles_tex_coord_edge1[hit_triangle_id], triangles_tex_coord_edge2[hit_triangle_id]);

		hit_normal = normalize(hit_normal);

		if (light_count > 0) {
			if (material.type == Material::Type::LIGHT) {
				if (last_specular) {
					return colour + throughput * material.emission;
				} else {
					return colour;
				}
			}

			if ((material.type == Material::Type::DIFFUSE) ||
				(material.type == Material::Type::GLOSSY && material.roughness >= ROUGHNESS_CUTOFF)) {

				// Pick a random light emitting triangle
				int light_triangle_id = light_indices[rand_xorshift(seed) % light_count];
				int light_material_id = triangles_material_id[light_triangle_id];

				ASSERT(length(materials[light_material_id].emission) > 0.0f, "Material was not emissive!\n");
			
				// Pick a random point on the triangle using random barycentric coordinates
				float u = random_float(seed);
				float v = random_float(seed);

				if (u + v > 1.0f) {
					u = 1.0f - u;
					v = 1.0f - v;
				}

				float3 light_point  = barycentric(u, v, triangles_position0[light_triangle_id], triangles_position_edge1[light_triangle_id], triangles_position_edge2[light_triangle_id]);
				float3 light_normal = barycentric(u, v, triangles_normal0  [light_triangle_id], triangles_normal_edge1  [light_triangle_id], triangles_normal_edge2  [light_triangle_id]);
				
				light_normal = normalize(light_normal);
				
				// Calculate the area of the triangle light
				float light_area = 0.5f * length(cross(
					triangles_position_edge1[light_triangle_id], 
					triangles_position_edge2[light_triangle_id]
				));

				float3 to_light = light_point - hit_point;
				float distance_to_light_squared = dot(to_light, to_light);
				float distance_to_light         = sqrtf(distance_to_light_squared);

				// Normalize the vector to the light
				to_light /= distance_to_light;

				float cos_o = -dot(to_light, light_normal);
				float cos_i =  dot(to_light, hit_normal);

				if (cos_o > 0.0f && cos_i > 0.0f) {
					ray.origin    = hit_point;
					ray.direction = to_light;
					ray.direction_inv = make_float3(
						1.0f / ray.direction.x, 
						1.0f / ray.direction.y, 
						1.0f / ray.direction.z
					);

					// Check if the light is obstructed by any other object in the scene
					if (!mbvh_intersect(ray, distance_to_light - EPSILON)) {
						float3 brdf = material.albedo(hit_tex_coord.x, hit_tex_coord.y) * ONE_OVER_PI;
						float solid_angle = (cos_o * light_area) / distance_to_light_squared;

						float3 light_colour = materials[light_material_id].emission;

						colour += throughput * brdf * light_count * light_colour * solid_angle * cos_i;
					}
				}
			}
		}

		// Russian Roulette termination
		float p_survive = clamp(fmaxf(throughput.x, fmaxf(throughput.y, throughput.z)), 0.0f, 1.0f);
		if (random_float(seed) > p_survive) {
			return colour;
		}

		throughput /= p_survive;

		float3 direction;
		if (material.type == Material::Type::DIFFUSE) {
			throughput *= material.albedo(hit_tex_coord.x, hit_tex_coord.y);

			direction = cosine_weighted_diffuse_reflection(seed, hit_normal);

			last_specular = false;
		} else if (material.type == Material::Type::DIELECTRIC) {
			float3 direction_reflected = reflect(ray.direction, hit_normal);

			float3 normal;
			float  cos_theta;
		
			float n_1;
			float n_2;
		
			float dir_dot_normal = dot(ray.direction, hit_normal);
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
				float3 direction_refracted = normalize(eta * ray.direction + (eta * cos_theta - sqrtf(k)) * hit_normal);
		
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

			last_specular = true;
		} else if (material.type == Material::Type::GLOSSY) {
			float3 direction_in = -ray.direction;
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
		
			direction = reflect(ray.direction, micro_normal_world);
		
			float i_dot_m = dot(direction_in, micro_normal_world);
			float i_dot_n = dot(direction_in,       hit_normal);
			float o_dot_n = dot(direction,          hit_normal);
			float m_dot_n = dot(micro_normal_world, hit_normal);
		
			float D = beckmann_D(m_dot_n, alpha);
			float G = 
				beckmann_G1(i_dot_n, m_dot_n, alpha) * 
				beckmann_G1(o_dot_n, m_dot_n, alpha);
			float weight = abs(i_dot_m) * G / abs(i_dot_n * m_dot_n);

			throughput *= material.albedo(hit_tex_coord.x, hit_tex_coord.y) * weight;

			last_specular = material.roughness < ROUGHNESS_CUTOFF;
		}

		ray.origin    = hit_point;
		ray.direction = direction;
		ray.direction_inv = make_float3(
			1.0f / ray.direction.x, 
			1.0f / ray.direction.y, 
			1.0f / ray.direction.z
		);
	}

	return make_float3(0.0f);
}

__device__ float3 camera_position;
__device__ float3 camera_top_left_corner;
__device__ float3 camera_x_axis;
__device__ float3 camera_y_axis;

extern "C" __global__ void trace_ray(int frame_number, float frames_since_camera_moved) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int thread_id = x + y * SCREEN_WIDTH;

	unsigned seed = (thread_id + frame_number * 312080213) * 781939187;
	
	// Add random value between 0 and 1 so that after averaging we get anti-aliasing
	float u = x + random_float(seed);
	float v = y + random_float(seed);

	// Create primary Ray that starts at the Camera's position and goes trough the current pixel
	Ray ray;
	ray.origin    = camera_position;
	ray.direction = normalize(camera_top_left_corner
		+ u * camera_x_axis
		+ v * camera_y_axis
	);
	ray.direction_inv = make_float3(
		1.0f / ray.direction.x, 
		1.0f / ray.direction.y, 
		1.0f / ray.direction.z
	);
	
	float3 colour = sample(seed, ray);

	// If the Camera hasn't moved, average over previous frames
	if (frames_since_camera_moved > 0.0f) {
		float4 prev;
		surf2Dread<float4>(&prev, frame_buffer, x * sizeof(float4), y);

		// Take average over n samples by weighing the current content of the framebuffer by (n-1) and the new sample by 1
		colour = (make_float3(prev) * (frames_since_camera_moved - 1.0f) + colour) / frames_since_camera_moved;
	}

	surf2Dwrite<float4>(make_float4(colour, 1.0f), frame_buffer, x * sizeof(float4), y, cudaBoundaryModeClamp);
}
