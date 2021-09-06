#pragma once
#include "Util.h"

__device__ __constant__ float2 * pmj_samples;
__device__ __constant__ uchar2 * blue_noise_textures;

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
	unsigned hash = wang_hash((pixel_index * unsigned(SampleDimension::NUM_DIMENSIONS) + unsigned(Dim)) * MAX_BOUNCES + bounce);

	// If we run out of PMJ02 samples, fall back to random
	if (sample_index >= PMJ_NUM_SAMPLES_PER_SEQUENCE) {
		float x = hash_with(sample_index,              hash) * (1.0f / float(0xffffffff));
		float y = hash_with(sample_index + 0xdeadbeef, hash) * (1.0f / float(0xffffffff));

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
__device__ float3 sample_ggx_distribution_of_normals(const float3 & omega, float alpha_x, float alpha_y, float u1, float u2){
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
	t2 = (1.0f - s) * sqrtf(1.0 - t1*t1) + s*t2;

	// Reproject onto hemisphere
	float3 n_h = t1*axis_1 + t2*axis_2 + sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2)) * v;

	// Transform the normal back to the ellipsoid configuration
	return normalize(make_float3(alpha_x * n_h.x, alpha_y * n_h.y, fmaxf(0.0f, n_h.z)));
}

__device__ int sample_light(float u1, float u2, int & transform_id) {
	// Pick random light emitting Mesh based on power
	float r1 = u1 * lights_total_power;

	int   light_mesh_id = 0;
	float light_power = light_mesh_power_scaled[0];

	while (r1 > light_power) {
		light_power += light_mesh_power_scaled[++light_mesh_id];
	}

	transform_id = light_mesh_transform_indices[light_mesh_id];

	// Pick random light emitting Triangle on the Mesh based on power
	int triangle_first_index = light_mesh_triangle_first_index[light_mesh_id];
	int triangle_count       = light_mesh_triangle_count      [light_mesh_id];

	int index_first = triangle_first_index;
	int index_last  = triangle_first_index + triangle_count - 1;

	float r2 = u2 * light_mesh_power_unscaled[light_mesh_id];
	return light_indices[binary_search(light_power_cumulative, index_first, index_last, r2)];
}
