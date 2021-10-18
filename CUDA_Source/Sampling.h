#pragma once
#include "Util.h"

__device__ __constant__ float2 * pmj_samples;
__device__ __constant__ uchar2 * blue_noise_textures;

__device__ __constant__ float lights_total_weight;

struct alignas(float2) ProbAlias {
	float prob;
	int   alias;
};

__device__ __constant__ const int       * light_indices;
__device__ __constant__ const ProbAlias * light_prob_alias;

__device__ __constant__ int               light_mesh_count;
__device__ __constant__ const ProbAlias * light_mesh_prob_alias;
__device__ __constant__ const int2      * light_mesh_first_index_and_triangle_count;
__device__ __constant__ const int       * light_mesh_transform_index;

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
	BRDF,

	NUM_DIMENSIONS,
	NUM_BOUNCE = 4 // Last four dimensions are reused every bounce
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
		return sqrtf(2.0f * u) - 1.0f;
	} else {
		return 1.0f - sqrtf(2.0f - 2.0f * u);
	}
}

// Box-Muller transform
__device__ float2 sample_gaussian(float u1, float u2) {
	float f = sqrt(-2.0f * logf(u1));
	float a = TWO_PI * u2;

	float sin_a, cos_a;
	__sincosf(a, &sin_a, &cos_a);

	return make_float2(f * cos_a, f * sin_a);
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
		phi = (0.25f * PI) * (b/a);
	} else {
		r = b;
		phi = (0.25f * PI) * (a/b) + (0.5f * PI);
	}

	float sin_phi, cos_phi;
	__sincosf(phi, &sin_phi, &cos_phi);

	return make_float2(r * cos_phi, r * sin_phi);
}

__device__ float3 sample_cosine_weighted_direction(float u1, float u2) {
	float2 d = sample_disk(u1, u2);
	return make_float3(d.x, d.y, sqrtf(1.0f - dot(d, d)));
}

// Based on: Heitz - Sampling the GGX Distribution of Visible Normals
__device__ float3 sample_visible_normals_ggx(const float3 & omega, float alpha_x, float alpha_y, float u1, float u2){
	// Transform the view direction to the hemisphere configuration
	float3 v = normalize(make_float3(alpha_x * omega.x, alpha_y * omega.y, omega.z));

	// Orthonormal basis (with special case if cross product is zero)
	float length_squared = v.x*v.x + v.y*v.y;
	float3 axis_1 = length_squared > 0.0f ? make_float3(-v.y, v.x, 0.0f) / sqrtf(length_squared) : make_float3(1.0f, 0.0f, 0.0f);
	float3 axis_2 = cross(v, axis_1);

	// Parameterization of the projected area
	float2 d = sample_disk(u1, u2);
	float t1 = d.x;
	float t2 = d.y;

	float s = 0.5f * (1.0f + v.z);
	t2 = (1.0f - s) * sqrtf(1.0f - t1*t1) + s*t2;

	// Reproject onto hemisphere
	float3 n_h = t1*axis_1 + t2*axis_2 + safe_sqrt(1.0f - t1*t1 - t2*t2) * v;

	// Transform the normal back to the ellipsoid configuration
	return normalize(make_float3(alpha_x * n_h.x, alpha_y * n_h.y, n_h.z));
}

// Draw sample from arbitrary distribution in O(1) time using the alias method
// Based on: in Vose - A Linear Algorithm for Generating Random Numbers with a Given Distribution (1991)
__device__ int sample_alias_method(float u, const ProbAlias * distribution, int n) {
	assert(u < 1.0f);
	u *= float(n);

	float u_fract = u - floorf(u);
	int   j       = __float2int_rd(u);

	ProbAlias prob_alias = distribution[j];

	// Choose j according to probability prob using the fractional part of u, otherwise choose the alias index
	if (u_fract <= prob_alias.prob) {
		return j;
	} else {
		return prob_alias.alias;
	}
}

__device__ int sample_light(float u1, float u2, int & transform_id) {
	// Pick random light emitting Mesh
	int light_mesh_id = sample_alias_method(u1, light_mesh_prob_alias, light_mesh_count);
	transform_id = light_mesh_transform_index[light_mesh_id];

	// Pick random light emitting Triangle on the Mesh
	int2 first_index_and_triangle_count = light_mesh_first_index_and_triangle_count[light_mesh_id];
	int  first_index    = first_index_and_triangle_count.x;
	int  triangle_count = first_index_and_triangle_count.y;

	int light_triangle_id = first_index + sample_alias_method(u2, light_prob_alias + first_index, triangle_count);
	return light_indices[light_triangle_id];
}
