#pragma once
#include "Ray.h"

__device__ __constant__ float4 * frame_buffer_moment;

// GBuffers used by SVGF
__device__ __constant__ Surface<float4> gbuffer_normal_and_depth;
__device__ __constant__ Surface<int2>   gbuffer_mesh_id_and_triangle_id;
__device__ __constant__ Surface<float2> gbuffer_screen_position_prev;

// SVGF History Buffers (Temporally Integrated)
__device__ __constant__ int    * history_length;
__device__ __constant__ float4 * history_direct;
__device__ __constant__ float4 * history_indirect;
__device__ __constant__ float4 * history_moment;
__device__ __constant__ float4 * history_normal_and_depth;

// Used for TAA
__device__ __constant__ float4 * taa_frame_curr;
__device__ __constant__ float4 * taa_frame_prev;

// Vector3 buffer in SoA layout
struct Vector3_SoA {
	float * x;
	float * y;
	float * z;

	__device__ void set(int index, const float3 & vector) {
		x[index] = vector.x;
		y[index] = vector.y;
		z[index] = vector.z;
	}

	__device__ float3 get(int index) const {
		return make_float3(
			x[index],
			y[index],
			z[index]
		);
	}
};

struct HitBuffer {
	uint4 * hits;

	__device__ void set(int index, const RayHit & ray_hit) {
		unsigned uv = int(ray_hit.u * 65535.0f) | (int(ray_hit.v * 65535.0f) << 16);

		hits[index] = make_uint4(ray_hit.mesh_id, ray_hit.triangle_id, float_as_uint(ray_hit.t), uv);
	}

	__device__ RayHit get(int index) const {
		uint4 hit = __ldg(&hits[index]);

		RayHit ray_hit;

		ray_hit.mesh_id     = hit.x;
		ray_hit.triangle_id = hit.y;
		
		ray_hit.t = uint_as_float(hit.z);

		ray_hit.u = float(hit.w & 0xffff) / 65535.0f;
		ray_hit.v = float(hit.w >> 16)    / 65535.0f;

		return ray_hit;
	}
};

// Input to the Trace and Sort Kernels in SoA layout
struct TraceBuffer {
	Vector3_SoA origin;
	Vector3_SoA direction;

#if ENABLE_MIPMAPPING
	float2 * cone;
#endif

	HitBuffer hits;

	int       * pixel_index_and_mis_eligable; // mis_eligable is a single bit (most significant bit) that indicates the previous Material has a BRDF that supports MIS
	Vector3_SoA throughput;

	float * last_pdf;
};

// Input to the various Shade Kernels in SoA layout
struct MaterialBuffer {
	Vector3_SoA direction;	

#if ENABLE_MIPMAPPING
	float2 * cone;
#endif

	HitBuffer hits;

	int       * pixel_index;
	Vector3_SoA throughput;
};

// Input to the Shadow Trace Kernel in SoA layout
struct ShadowRayBuffer {
	Vector3_SoA ray_origin;
	Vector3_SoA ray_direction;

	float * max_distance;

	int       * pixel_index;
	Vector3_SoA illumination;
};

__device__ __constant__ TraceBuffer     ray_buffer_trace;
__device__ __constant__ MaterialBuffer  ray_buffer_shade_diffuse;
__device__ __constant__ MaterialBuffer  ray_buffer_shade_dielectric_and_glossy;
__device__ __constant__ ShadowRayBuffer ray_buffer_shadow;

// Number of elements in each Buffer
// Sizes are stored for ALL bounces so we only have to reset these
// values back to 0 after every frame, instead of after every bounce
struct BufferSizes {
	int trace     [MAX_BOUNCES];
	int diffuse   [MAX_BOUNCES];
	int dielectric[MAX_BOUNCES];
	int glossy    [MAX_BOUNCES];
	int shadow    [MAX_BOUNCES];

	// Global counters for tracing kernels
	int rays_retired       [MAX_BOUNCES];
	int rays_retired_shadow[MAX_BOUNCES];
} __device__ buffer_sizes;
