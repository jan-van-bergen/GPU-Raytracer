#pragma once
#include "Util.h"

__device__ __constant__ float2 * pmj_samples;

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

	unsigned index = permute(sample_index, PMJ_NUM_SAMPLES_PER_SEQUENCE, hash);
	unsigned dim = unsigned(Dim) + unsigned(SampleDimension::NUM_BOUNCE) * bounce;

	const float2 * pmj_sequence = pmj_samples + (dim % PMJ_NUM_SEQUENCES) * PMJ_NUM_SAMPLES_PER_SEQUENCE;
	return pmj_sequence[index];
}

__device__ float2 box_muller(float u1, float u2) {
	float f = sqrt(-2.0f * logf(u1));
	float a = TWO_PI * u2;
	
	float sin_a, cos_a;
	__sincosf(a, &sin_a, &cos_a);

	return make_float2(f * cos_a, f * sin_a);
}

// Based on: Heitz - A Low-Distortion Map Between Triangle and Square 
__device__ float2 sample_triangle(float u1, float u2) {
	float2 uv = make_float2(0.5f * u1, 0.5f * u2);
	float offset = uv.y - uv.x;
	if (offset > 0.0f) {
		uv.y += offset;
	} else {
		uv.x -= offset;
	}
	return uv;
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
