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
	// Only do this on the first bounce, because we use up the 8 available dimensions on the first bounce
    if (sample_index < 256 && bounce == 0) {
        return random_heitz(x, y, sample_index, dimension);
    } else {
        return random_float_xorshift(seed);
    }
}

__device__ float3 random_cosine_weighted_diffuse_reflection(int x, int y, int sample_index, int bounce, unsigned & seed, const float3 & normal) {
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

	ASSERT(dot(direction, normal) > -1e-5, "Invalid dot: dot = %f, direction = (%f, %f, %f), normal = (%f, %f, %f)\n", 
		dot(direction, normal), direction.x, direction.y, direction.z, normal.x, normal.y, normal.z
	);

	return direction;
}
