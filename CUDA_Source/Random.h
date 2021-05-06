#pragma once

__device__ __constant__ unsigned char * sobol_256spp_256d; // 256 * 256
__device__ __constant__ unsigned char * scrambling_tile;   // 128 * 128 * 8
__device__ __constant__ unsigned char * ranking_tile;      // 128 * 128 * 4

// Based on: A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space - Heitz et al. 19
__device__ float random_heitz(int x, int y, int sample_index, int sample_dimension) {
	x &= 127;
	y &= 127;
	sample_dimension &= 7;

	// xor index based on optimized ranking
	int ranked_sample_index = sample_index ^ ranking_tile[(sample_dimension + (x + y * 128) * 8) >> 1];

	// fetch value in sequence
	int value = sobol_256spp_256d[sample_dimension + ranked_sample_index * 256];

	// If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ scrambling_tile[sample_dimension + (x + y * 128) * 8];

	// convert to float and return
	return (value + 0.5f) * (1.0f / 256.0f);
}

__device__ unsigned wang_hash(unsigned seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);

    return seed;
}

// Based on: http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
__device__ unsigned random_xorshift(unsigned & seed) {
	seed ^= (seed << 13);
	seed ^= (seed >> 17);
	seed ^= (seed << 5);

	return seed;
}

__device__ float random_float_xorshift(unsigned & seed) {
	const float one_over_max_unsigned = 2.3283064365387e-10f;
	return float(random_xorshift(seed)) * one_over_max_unsigned;
}

__device__ float random_float_heitz(int x, int y, int sample_index, int bounce, int dimension, unsigned & seed) {
	// Use Blue Noise sampler for first 256 samples
	if (sample_index < 256) {
		return random_heitz(x, y, sample_index, (bounce * 8) + dimension);
	} else {
		return random_float_xorshift(seed);
	}
}

__device__ float2 box_muller(float u1, float u2) {
	float f = sqrt(-2.0f * logf(u1));
	float a = TWO_PI * u2;
	
	float sin_a, cos_a;
	__sincosf(a, &sin_a, &cos_a);

	return make_float2(f * cos_a, f * sin_a);
}

__device__ float3 random_cosine_weighted_direction(int x, int y, int sample_index, int bounce, unsigned & seed) {
	float r0 = random_float_heitz(x, y, sample_index, bounce, 2, seed);
	float r1 = random_float_heitz(x, y, sample_index, bounce, 3, seed);

	float sin_theta, cos_theta;
	sincos(TWO_PI * r1, &sin_theta, &cos_theta);

	float r = sqrtf(r0);
	float xf = r * cos_theta;
	float yf = r * sin_theta;
	
	return normalize(make_float3(xf, yf, sqrtf(1.0f - r0)));
}

__device__ int random_point_on_random_light(int x, int y, int sample_index, int bounce, unsigned & seed, float & u, float & v, int & transform_id) {
	// Pick random light emitting Mesh based on area
	float random_value = random_float_heitz(x, y, sample_index, bounce, 4, seed) * light_total_area;

	int   light_mesh_id = 0;
	float light_area_cumulative = light_mesh_area_scaled[0];

	while (random_value > light_area_cumulative) {
		light_area_cumulative += light_mesh_area_scaled[++light_mesh_id];
	}

	// Pick random light emitting Triangle on the Mesh based on area
	int triangle_first_index = light_mesh_triangle_first_index[light_mesh_id];
	int triangle_count       = light_mesh_triangle_count      [light_mesh_id];

	int index_left  = triangle_first_index;
	int index_right = triangle_first_index + triangle_count - 1;

	random_value = random_float_heitz(x, y, sample_index, bounce, 5, seed) * light_mesh_area_unscaled[light_mesh_id];

	// Binary search
	int light_triangle_id;
	while (true) {
		int index_middle = (index_left + index_right) / 2;

		if (index_middle > triangle_first_index && random_value <= light_areas_cumulative[index_middle - 1]) {
			index_right = index_middle - 1;
		} else if (random_value > light_areas_cumulative[index_middle]) {
			index_left = index_middle + 1;
		} else {
			light_triangle_id = light_indices[index_middle];

			break;
		}
	}

	// Pick a random point on the triangle using random barycentric coordinates
	u = random_float_heitz(x, y, sample_index, bounce, 6, seed);
	v = random_float_heitz(x, y, sample_index, bounce, 7, seed);

	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}

	transform_id = light_mesh_transform_indices[light_mesh_id];

	return light_triangle_id;
}
