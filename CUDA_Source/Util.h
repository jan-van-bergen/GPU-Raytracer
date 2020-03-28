#pragma once
#include <vector_types.h>

#include "cuda_math.h"

//#define ASSERT(proposition, fmt, ...) { if (!(proposition)) printf(fmt, __VA_ARGS__); assert(proposition); }
#define ASSERT(proposition, fmt, ...) { }

__device__ float luminance(float r, float g, float b) {
	return 0.299f * r + 0.587f * g + 0.114f * b;
}

template<typename T>
__device__ T barycentric(float u, float v, const T & base, const T & edge_1, const T & edge_2) {
	return base + u * edge_1 + v * edge_2;
}

__device__ void orthonormal_basis(const float3 & normal, float3 & tangent, float3 & binormal) {
	float sign = copysignf(1.0f, normal.z);
	float a = -1.0f / (sign + normal.z);
	float b = normal.x * normal.y * a;
	
	tangent  = make_float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
	binormal = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
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
