#pragma once

#if false
	#define ASSERT(proposition, fmt, ...) do { /*if (!(proposition)) printf(fmt, __VA_ARGS__);*/ assert(proposition); } while (false)
#else
	#define ASSERT(proposition, fmt, ...) do { } while (false)
#endif

// Type safe wrapper around Texture Object
template<typename T>
struct Texture {
	cudaTextureObject_t texture;

	// Width and height are required for Mipmap LOD calculations
	float2 size;

	__device__ inline T get(float s, float t) const {
		return tex2D<T>(texture, s, t);
	}

	__device__ inline T get_lod(float s, float t, float lod) const {
		return tex2DLod<T>(texture, s, t, lod);
	}

	__device__ inline T get_grad(float s, float t, float2 dx, float2 dy) const {
		return tex2DGrad<T>(texture, s, t, dx, dy);
	}
};

// Type safe wrapper around Surface Object
template<typename T>
struct Surface {
	cudaSurfaceObject_t surface;

	__device__ inline T get(int x, int y) const {
		// assert(x >= 0 && x < screen_width);
		// assert(y >= 0 && y < screen_height);
		
		T value;
		surf2Dread<T>(&value, surface, x * sizeof(T), y, cudaBoundaryModeClamp);
		return value;
	}

	__device__ inline void set(int x, int y, const T & value) {
		// assert(x >= 0 && x < screen_width);
		// assert(y >= 0 && y < screen_height);

		surf2Dwrite<T>(value, surface, x * sizeof(T), y);
	}
};

__device__ inline float luminance(float r, float g, float b) {
	return 0.299f * r + 0.587f * g + 0.114f * b;
}

__device__ inline float3 rgb_to_ycocg(const float3 & colour) {
	return make_float3(
		 0.25f * colour.x + 0.5f * colour.y + 0.25f * colour.z,
		 0.5f  * colour.x                   - 0.5f  * colour.z,
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

__device__ inline unsigned wang_hash(unsigned seed) {
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);

    return seed;
}

// Based on: https://github.com/blender/blender/blob/master/intern/cycles/kernel/kernel_jitter.h#L122
__device__ inline unsigned hash_with(unsigned i, unsigned p) {
	i = (i ^ 61) ^ p;
	i += i << 3;
	i ^= i >> 4;
	i *= 0x27d4eb2d;
	return i;
}

__device__ inline unsigned hash_combine(unsigned a, unsigned b) {
	return a ^ (b + 0x9e3779b9 + (a << 6) + (a >> 2));
}

// Based on: https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/util/math.h
__device__ inline unsigned permute(unsigned index, unsigned length, unsigned seed) {
	// NOTE: Assumes length is a power of two
	unsigned mask = length - 1;

	index ^= seed;
	index *= 0xe170893d;
	index ^= seed >> 16;
	index ^= (index & mask) >> 4;
	index ^= seed >> 8;
	index *= 0x0929eb3f;
	index ^= seed >> 23;
	index ^= (index & mask) >> 1;
	index *= 1 | seed >> 27;
	index *= 0x6935fa69;
	index ^= (index & mask) >> 11;
	index *= 0x74dcb303;
	index ^= (index & mask) >> 2;
	index *= 0x9e501cc3;
	index ^= (index & mask) >> 2;
	index *= 0xc860a3df;
	index &= mask;
	index ^= index >> 5;

	return (index + seed) & mask;
}

__device__ inline constexpr bool is_power_of_two(unsigned x) {
	if (x) {
		return (x & (x - 1)) == 0;
	} else {
		return false;
	}
}

template<typename T>
__device__ T lerp(T const & a, T const & b, float t) {
	return (1.0f - t) * a + t * b;
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
	int laneid = threadIdx.x & 31;
	
	int res;
	if (laneid == leader) {
		res = atomicAdd(ctr, __popc(mask));
	}

	res = __shfl_sync(mask, res, leader);
	return res + __popc(mask & ((1 << laneid) - 1));
}

// Based on: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
__device__ inline float2 oct_encode_normal(float3 n) {
	n /= (abs(n.x) + abs(n.y) + abs(n.z));

	if (n.z < 0.0f) {
		// Oct wrap
		n.x = (1.0f - abs(n.y)) * (n.x >= 0.0f ? +1.0f : -1.0f);	
		n.y = (1.0f - abs(n.x)) * (n.y >= 0.0f ? +1.0f : -1.0f);
	}

	return make_float2(0.5f + 0.5f * n.x, 0.5f + 0.5f * n.y);
}

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

// Binary search a cumulative (monotone increasing) array for the first index that is smaller than a given value
__device__ inline int binary_search(const float cumulative_array[], int index_first, int index_last, float value) {
	int index_left  = index_first;
	int index_right = index_last;

	while (true) {
		int index_middle = (index_left + index_right) / 2;

		if (index_middle > index_first && value <= cumulative_array[index_middle - 1]) {
			index_right = index_middle - 1;
		} else if (value > cumulative_array[index_middle]) {
			index_left = index_middle + 1;
		} else {
			return index_middle;
		}
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

// VMIN, VMAX functions, see "Understanding the Efficiency of Ray Traversal on GPUs – Kepler and Fermi Addendum" by Aila et al.

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
