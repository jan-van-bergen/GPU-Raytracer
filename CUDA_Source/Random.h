#pragma once

__device__ unsigned char * sobol_256spp_256d; // 256 * 256
__device__ unsigned char * scrambling_tile;   // 128 * 128 * 8
__device__ unsigned char * ranking_tile;      // 128 * 128 * 4

// Based on: A Low-Discrepancy Sampler that Distributes Monte Carlo Errors as a Blue Noise in Screen Space - Heitz et al. 19
__device__ float random_heitz(int x, int y, int sample_index, int sample_dimension) {
	x &= 127;
	y &= 127;
	
	// xor index based on optimized ranking
	int ranked_sample_index = sample_index ^ ranking_tile[(sample_dimension + (x + y * 128) * 8) >> 1];

	// fetch value in sequence
	int value = sobol_256spp_256d[sample_dimension + ranked_sample_index * 256];

	// If the dimension is optimized, xor sequence value based on optimized scrambling
	value = value ^ scrambling_tile[(sample_dimension & 7) + (x + y * 128) * 8];

	// convert to float and return
	return (value + 0.5f) * (1.0f / 256.0f);
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
		return random_heitz(x, y, sample_index, (bounce << 3) + dimension);
	} else {
		return random_float_xorshift(seed);
	}
}

__device__ float3 random_cosine_weighted_direction(int x, int y, int sample_index, int bounce, unsigned & seed, const float3 & normal) {
	float r0 = random_float_heitz(x, y, sample_index, bounce, 2, seed);
	float r1 = random_float_heitz(x, y, sample_index, bounce, 3, seed);

	float sin_theta, cos_theta;
	sincos(TWO_PI * r1, &sin_theta, &cos_theta);

	float r = sqrtf(r0);
	float xf = r * cos_theta;
	float yf = r * sin_theta;
	
	float3 direction = normalize(make_float3(xf, yf, sqrtf(1.0f - r0)));
	
	float3 tangent, binormal;
	orthonormal_basis(normal, tangent, binormal);

	// Multiply the direction with the TBN matrix
	direction = local_to_world(direction, tangent, binormal, normal);

	return direction;
}

__device__ int random_point_on_random_light(int x, int y, int sample_index, int bounce, unsigned & seed, float & u, float & v) {
#if LIGHT_SELECTION == LIGHT_SELECT_UNIFORM
	int light_triangle_id = light_indices[random_xorshift(seed) % light_count];
#elif LIGHT_SELECTION == LIGHT_SELECT_AREA
	// Pick random value between 0 and light_area_total
	float random_value = random_float_xorshift(seed) * light_area_total;

	int index_left  = 0;
	int index_right = light_count - 1;

	int light_triangle_id;
	while (true) {
		int index_middle = (index_left + index_right) >> 1;

		if (random_value < light_areas_cumulative[index_middle]) {
			index_right = index_middle - 1;
		} else if (random_value > light_areas_cumulative[index_middle + 1]) {
			index_left = index_middle + 1;
		} else {
			light_triangle_id = light_indices[index_middle];

			break;
		}
	}
#endif

	// Pick a random point on the triangle using random barycentric coordinates
	u = random_float_heitz(x, y, sample_index, bounce, 6, seed);
	v = random_float_heitz(x, y, sample_index, bounce, 7, seed);

	if (u + v > 1.0f) {
		u = 1.0f - u;
		v = 1.0f - v;
	}

	return light_triangle_id;
}
