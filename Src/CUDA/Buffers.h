#pragma once
#include "Raytracing/Ray.h"

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

		hits[index] = make_uint4(ray_hit.mesh_id, ray_hit.triangle_id, __float_as_uint(ray_hit.t), uv);
	}

	__device__ RayHit get(int index) const {
		uint4 hit = __ldg(&hits[index]);

		RayHit ray_hit;

		ray_hit.mesh_id     = hit.x;
		ray_hit.triangle_id = hit.y;

		ray_hit.t = __uint_as_float(hit.z);

		ray_hit.u = float(hit.w & 0xffff) / 65535.0f;
		ray_hit.v = float(hit.w >> 16)    / 65535.0f;

		return ray_hit;
	}
};
