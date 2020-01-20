#pragma once
#include <vector_types.h>

#include "cuda_math.h"

//#define ASSERT(proposition, fmt, ...) { if (!(proposition)) printf(fmt, __VA_ARGS__); assert(proposition); }
#define ASSERT(proposition, fmt, ...) { }

// Based on: http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
__device__ unsigned rand_xorshift(unsigned & seed) {
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
	seed ^= (seed << 5);
	
    return seed;
}

__device__ float random_float(unsigned & seed) {
	const float one_over_max_unsigned = 2.3283064365387e-10f;
	return float(rand_xorshift(seed)) * one_over_max_unsigned;
}

template<typename T>
__device__ T barycentric(float u, float v, const T & base, const T & edge_1, const T & edge_2) {
	return base + u * edge_1 + v * edge_2;
}

__device__ void orthonormal_basis(const float3 & normal, float3 & tangent, float3 & binormal) {
	// Calculate a tangent vector from the normal vector
	if (fabsf(normal.x) > 0.99f) {
		tangent = make_float3(-normal.z, 0.0f, normal.x) * rsqrt(normal.x * normal.x + normal.z * normal.z);
	} else {
		tangent = make_float3(0.0f, normal.z, -normal.y) * rsqrt(normal.y * normal.y + normal.z * normal.z);
	}

	// The binormal is perpendicular to both the normal and tangent vectors
	binormal = cross(normal, tangent);
}

__device__ float3 local_to_world(const float3 & vector, const float3 & tangent, const float3 & binormal, const float3 & normal) {
	return make_float3(
		tangent.x * vector.x + binormal.x * vector.y + normal.x * vector.z, 
		tangent.y * vector.x + binormal.y * vector.y + normal.y * vector.z, 
		tangent.z * vector.x + binormal.z * vector.y + normal.z * vector.z
	);
}

__device__ float3 world_to_local(const float3 & vector, const float3 & tangent, const float3 & binormal, const float3 & normal) {
	return make_float3(dot(tangent, vector), dot(binormal, vector), dot(normal, vector));
}

__device__ float3 cosine_weighted_diffuse_reflection(unsigned & seed, const float3 & normal) {
	float r0 = random_float(seed);
	float r1 = random_float(seed);

	float sin_theta, cos_theta;
	sincos(TWO_PI * r1, &sin_theta, &cos_theta);

	float r = sqrtf(r0);
	float x = r * cos_theta;
	float y = r * sin_theta;
	
	float3 direction = normalize(make_float3(x, y, sqrtf(1.0f - r0)));
	
	float3 tangent, binormal;
	orthonormal_basis(normal, tangent, binormal);

	// Multiply the direction with the TBN matrix
	direction = local_to_world(direction, tangent, binormal, normal);

	ASSERT(dot(direction, normal) > -1e-5, "Invalid dot: dot = %f, direction = (%f, %f, %f), normal = (%f, %f, %f)\n", 
		dot(direction, normal), direction.x, direction.y, direction.z, normal.x, normal.y, normal.z
	);

	return direction;
}

// Based on: https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ inline int atomic_agg_inc(int * ctr) {
	int mask   = __ballot(1);
	int leader = __ffs(mask) - 1;
	int laneid = threadIdx.x % 32;
	
	int res;
	if (laneid == leader) {
		res = atomicAdd(ctr, __popc(mask));
	}

	res = __shfl(res, leader);
	return res + __popc(mask & ((1 << laneid) - 1));
}
