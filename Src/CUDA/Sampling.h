#pragma once
#include "Util.h"
#include "Config.h"

__device__ __constant__ float2 * pmj_samples;
__device__ __constant__ uchar2 * blue_noise_textures;

__device__ __constant__ float lights_total_weight;

__device__ __constant__ const int   * light_triangle_indices;
__device__ __constant__ const float * light_triangle_cumulative_probability;

__device__ __constant__ int           light_mesh_count;
__device__ __constant__ const float * light_mesh_cumulative_probability;
__device__ __constant__ const int2  * light_mesh_triangle_span; // First and last index into 'light_mesh_area_cumulative' array
__device__ __constant__ const int   * light_mesh_transform_indices;

__device__ inline bool pdf_is_valid(float pdf) {
	return isfinite(pdf) && pdf > 1e-4f;
}

__device__ inline float balance_heuristic(float pdf_f, float pdf_g) {
	return pdf_f / (pdf_f + pdf_g);
}

__device__ inline float power_heuristic(float pdf_f, float pdf_g) {
	return (pdf_f * pdf_f) / (pdf_f * pdf_f + pdf_g * pdf_g); // Power of 2 hardcoded, best empirical results according to Veach
}

enum struct SampleDimension {
	FILTER,
	APERTURE,

	RUSSIAN_ROULETTE,
	NEE_LIGHT,
	NEE_TRIANGLE,
	BSDF_0,
	BSDF_1,

	NUM_DIMENSIONS,
	NUM_BOUNCE = 5 // Last 5 dimensions are reused every bounce
};

template<SampleDimension Dim>
__device__ float2 random(unsigned pixel_index, unsigned bounce, unsigned sample_index) {
	unsigned hash = pcg_hash((pixel_index * unsigned(SampleDimension::NUM_DIMENSIONS) + unsigned(Dim)) * MAX_BOUNCES + bounce);

	// If we run out of PMJ02 samples, fall back to random
	if (sample_index >= PMJ_NUM_SAMPLES_PER_SEQUENCE) {
		const float one_over_max_unsigned = __uint_as_float(0x2f7fffff); // Constant such that 0xffffffff will map to a float strictly less than 1.0f

		float x = hash_with(sample_index,              hash) * one_over_max_unsigned;
		float y = hash_with(sample_index + 0xdeadbeef, hash) * one_over_max_unsigned;

		return make_float2(x, y);
	}

	unsigned dim = unsigned(Dim) + unsigned(SampleDimension::NUM_BOUNCE) * bounce;

	// If we run out of unique PMJ sequences, reuse a previous one but permute the index
	if (dim >= PMJ_NUM_SEQUENCES) {
		sample_index = permute(sample_index, PMJ_NUM_SAMPLES_PER_SEQUENCE, hash);
	}

	const float2 * pmj_sequence = pmj_samples + (dim % PMJ_NUM_SEQUENCES) * PMJ_NUM_SAMPLES_PER_SEQUENCE;
	float2 sample = pmj_sequence[sample_index];

	// Apply Cranley-Patterson rotation
	uchar2 * blue_noise_texture = blue_noise_textures + (dim % BLUE_NOISE_NUM_TEXTURES) * (BLUE_NOISE_TEXTURE_DIM * BLUE_NOISE_TEXTURE_DIM);

	int x = (pixel_index % screen_pitch) % BLUE_NOISE_TEXTURE_DIM;
	int y = (pixel_index / screen_pitch) % BLUE_NOISE_TEXTURE_DIM;

	uchar2 blue_noise = blue_noise_texture[x + y * BLUE_NOISE_TEXTURE_DIM];
	sample += make_float2(
		blue_noise.x * (1.0f / 255.0f),
		blue_noise.y * (1.0f / 255.0f)
	);

	if (sample.x >= 1.0f) sample.x -= 1.0f;
	if (sample.y >= 1.0f) sample.y -= 1.0f;

	return sample;
}

__device__ float sample_tent(float u) {
	if (u < 0.5f) {
		return safe_sqrt(2.0f * u) - 1.0f;
	} else {
		return 1.0f - safe_sqrt(2.0f - 2.0f * u);
	}
}

// Box-Muller transform
__device__ float2 sample_gaussian(float u1, float u2) {
	float f = sqrt(-2.0f * logf(u1));
	float a = TWO_PI * u2;
	return f * sincos(a);
}

__device__ float sample_exp(float lambda, float u) {
	return -logf(u) / lambda;
}

// Based on: Heitz - A Low-Distortion Map Between Triangle and Square
__device__ float2 sample_triangle(float u1, float u2) {
	if (u2 > u1) {
		u1 *= 0.5f;
		u2 -= u1;
	} else {
		u2 *= 0.5f;
		u1 -= u2;
	}
	return make_float2(u1, u2);
}

// Based on: http://psgraphics.blogspot.com/2011/01/improved-code-for-concentric-map.html
__device__ float2 sample_disk(float u1, float u2) {
	float a = 2.0f * u1 - 1.0f;
	float b = 2.0f * u2 - 1.0f;

	float phi, r;
	if (a*a > b*b) {
		r = a;
		phi = 0.25f*PI * (b/a);
	} else {
		r = b;
		phi = 0.5f*PI - 0.25f*PI * (a/b);
	}

	return r * sincos(phi);
}

__device__ float3 sample_cosine_weighted_direction(float u1, float u2) {
	float2 d = sample_disk(u1, u2);
	return make_float3(d.x, d.y, safe_sqrt(1.0f - dot(d, d)));
}

// Based on PBRT v3
__device__ float3 sample_henyey_greenstein(float3 omega, float g, float u1, float u2) {
	float cos_theta;
	if (fabsf(g) < 1e-3f) {
		// Isotropic case
		cos_theta = 1.0f - 2.0f * u1;
	} else {
		cos_theta = -(1.0f + g * g - square((1.0f - g * g) / (1.0f + g - 2.0f * g * u1))) / (2.0f * g);
	}
	float sin_theta = safe_sqrt(1.0f - square(cos_theta));
	float2 sincos_phi = sincos(TWO_PI * u2);
	float3 direction = spherical_to_cartesian(sin_theta, cos_theta, sincos_phi.x, sincos_phi.y);

	float3 v1, v2;
	orthonormal_basis(omega, v1, v2);

	return local_to_world(direction, v1, v2, omega);
}

// Based on: Heitz - Sampling the GGX Distribution of Visible Normals
__device__ float3 sample_visible_normals_ggx(float3 omega, float alpha_x, float alpha_y, float u1, float u2){
	// Transform the view direction to the hemisphere configuration
	float3 v = normalize(make_float3(alpha_x * omega.x, alpha_y * omega.y, omega.z));

	// Orthonormal basis (with special case if cross product is zero)
	float length_squared = v.x*v.x + v.y*v.y;
	float3 axis_1 = length_squared > 0.0f ? make_float3(-v.y, v.x, 0.0f) / sqrtf(length_squared) : make_float3(1.0f, 0.0f, 0.0f);
	float3 axis_2 = cross(v, axis_1);

	// Parameterization of the projected area
	float2 d = sample_disk(u1, u2);
	float t1 = d.x;
	float t2 = lerp(safe_sqrt(1.0f - t1*t1), d.y, 0.5f + 0.5f * v.z);

	// Reproject onto hemisphere
	float3 n_h = t1*axis_1 + t2*axis_2 + safe_sqrt(1.0f - t1*t1 - t2*t2) * v;

	// Transform the normal back to the ellipsoid configuration
	return normalize(make_float3(alpha_x * n_h.x, alpha_y * n_h.y, n_h.z));
}

__device__ int sample_light(float u1, float u2, int & transform_id) {
	// Pick light emitting Mesh
	int light_mesh_id = binary_search(light_mesh_cumulative_probability, 0, light_mesh_count - 1, u1);
	transform_id = light_mesh_transform_indices[light_mesh_id];

	// Pick light emitting Triangle on the Mesh
	int2 triangle_span = light_mesh_triangle_span[light_mesh_id];
	int light_triangle_id = binary_search(light_triangle_cumulative_probability, triangle_span.x, triangle_span.y, u2);

	return light_triangle_indices[light_triangle_id];
}
