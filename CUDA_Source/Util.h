#pragma once
#include <vector_types.h>

#include "cuda_math.h"

//#define ASSERT(proposition, fmt, ...) do { if (!(proposition)) printf(fmt, __VA_ARGS__); assert(proposition); } while(false)
#define ASSERT(proposition, fmt, ...) { }

__device__ inline float luminance(float r, float g, float b) {
	return 0.299f * r + 0.587f * g + 0.114f * b;
}

__device__ inline float3 rgb_to_ycocg(const float3 & colour) {
	return make_float3(
		 0.25f * colour.x + 0.5f * colour.y + 0.25f * colour.z,
		 0.5f  * colour.x +                 - 0.5f  * colour.z,
		-0.25f * colour.x + 0.5f * colour.y - 0.25f * colour.z
	);
}

__device__ inline float3 ycocg_to_rgb(const float3 & colour) {
	return make_float3(
		saturate(colour.x + colour.y - colour.z),
		saturate(colour.x            + colour.z),
		saturate(colour.x - colour.y - colour.z)
	);
}

template<typename T>
__device__ inline T barycentric(float u, float v, const T & base, const T & edge_1, const T & edge_2) {
	return base + u * edge_1 + v * edge_2;
}

__device__ inline void orthonormal_basis(const float3 & normal, float3 & tangent, float3 & binormal) {
	float sign = copysignf(1.0f, normal.z);
	float a = -1.0f / (sign + normal.z);
	float b = normal.x * normal.y * a;
	
	tangent  = make_float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
	binormal = make_float3(b, sign + normal.y * normal.y * a, -normal.y);
}

__device__ inline float3 local_to_world(const float3 & vector, const float3 & tangent, const float3 & binormal, const float3 & normal) {
	return make_float3(
		tangent.x * vector.x + binormal.x * vector.y + normal.x * vector.z, 
		tangent.y * vector.x + binormal.y * vector.y + normal.y * vector.z, 
		tangent.z * vector.x + binormal.z * vector.y + normal.z * vector.z
	);
}

__device__ inline float3 world_to_local(const float3 & vector, const float3 & tangent, const float3 & binormal, const float3 & normal) {
	return make_float3(dot(tangent, vector), dot(binormal, vector), dot(normal, vector));
}

__device__ inline unsigned active_thread_mask() {
	return __ballot_sync(0xffffffff, 1);
}

// Based on: https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ inline int atomic_agg_inc(int * ctr) {
	int mask   = active_thread_mask();
	int leader = __ffs(mask) - 1;
	int laneid = threadIdx.x % 32;
	
	int res;
	if (laneid == leader) {
		res = atomicAdd(ctr, __popc(mask));
	}

	res = __shfl_sync(mask, res, leader);
	return res + __popc(mask & ((1 << laneid) - 1));
}

// Based on: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
__device__ inline float3 oct_decode_normal(float2 f) {
	f = f * 2.0f - 1.0f;

	float3 n = make_float3(f.x, f.y, 1.0f - fabsf(f.x) - fabsf(f.y));

	float t = saturate(-n.z);
	n.x += n.x >= 0.0 ? -t : t;
	n.y += n.y >= 0.0 ? -t : t;

	return normalize(n);
}

__device__ float mitchell_netravali(float x) {
	const float B = 1.0f / 3.0f;
	const float C = 1.0f / 3.0f;

	x = fabsf(x);
	float x2 = x  * x;
	float x3 = x2 * x;

	if (x < 1.0f) {
		return (1.0f / 6.0f) * ((12.0f - 9.0f * B - 6.0f * C) * x3 + (-18.0f + 12.0f * B + 6.0f  * C) * x2 + (6.0f - 2.0f * B));
	} else if (x < 2.0f) {
		return (1.0f / 6.0f) * (              (-B - 6.0f * C) * x3 +           (6.0f * B + 30.0f * C) * x2 + (-12.0f * B - 48.0f * C) * x + (8.0f * B + 24.0f * C)); 
	} else {
		return 0.0f;
	}
}

// Create byte mask from sign bit
__device__ unsigned sign_extend_s8x4(unsigned x) {
	unsigned result;
	asm("prmt.b32 %0, %1, 0x0, 0x0000BA98;" : "=r"(result) : "r"(x));
	return result;
}

// Most significant bit
__device__ unsigned msb(unsigned x) {
	unsigned result;
	asm volatile("bfind.u32 %0, %1; " : "=r"(result) : "r"(x));
	return result;
}

// Extracts the i-th most significant byte from x
__device__ unsigned extract_byte(unsigned x, unsigned i) {
	return (x >> (i * 8)) & 0xff;
}

// VMIN, VMAX functions, see "Understanding the Efficiency of Ray Traversal on GPUs –Kepler and Fermi Addendum" by Aila et al.

// Computes min(min(a, b), c)
__device__ float vmin_min(float a, float b, float c) {
	int result;

	asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));
	
	return __int_as_float(result);
}

// Computes max(min(a, b), c)
__device__ float vmin_max(float a, float b, float c) {
	int result;

	asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));
	
	return __int_as_float(result);
}

// Computes min(max(a, b), c)
__device__ float vmax_min(float a, float b, float c) {
	int result;
	
	asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));
	
	return __int_as_float(result);
}

// Computes max(max(a, b), c)
__device__ float vmax_max(float a, float b, float c) {
	int result;

	asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(result) : "r"(__float_as_int(a)), "r"(__float_as_int(b)), "r"(__float_as_int(c)));
	
	return __int_as_float(result);
}
