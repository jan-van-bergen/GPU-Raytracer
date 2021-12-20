#pragma once

struct RayHit {
	float t;
	float u, v;

	int mesh_id;
	int triangle_id;
};

struct Ray {
	float3 origin;
	float3 direction;
	float3 direction_inv;

	__device__ inline void calc_direction_inv() {
		direction_inv = make_float3(
			1.0f / direction.x,
			1.0f / direction.y,
			1.0f / direction.z
		);
	}
};

// Based on: Wächter and Binder - A Fast and Robust Method for Avoiding Self-Intersection
__device__ float3 ray_origin_epsilon_offset(const float3 & origin, const float3 & direction, const float3 & geometric_normal) {
	constexpr static float ORIGIN      = 1.0f / 32.0f;
	constexpr static float FLOAT_SCALE = 1.0f / 65536.0f;
	constexpr static float INT_SCALE   = 256.0f;

	float3 n = sign(dot(direction, geometric_normal)) * geometric_normal;

	int3 of_i = make_int3(
		int(INT_SCALE * n.x),
		int(INT_SCALE * n.y),
		int(INT_SCALE * n.z)
	);
	float3 p_i = make_float3(
		__int_as_float(__float_as_int(origin.x) + ((origin.x < 0.0f) ? -of_i.x : of_i.x)),
		__int_as_float(__float_as_int(origin.y) + ((origin.y < 0.0f) ? -of_i.y : of_i.y)),
		__int_as_float(__float_as_int(origin.z) + ((origin.z < 0.0f) ? -of_i.z : of_i.z))
	);

	return make_float3(
		fabsf(origin.x) < ORIGIN ? origin.x + FLOAT_SCALE * n.x : p_i.x,
		fabsf(origin.y) < ORIGIN ? origin.y + FLOAT_SCALE * n.y : p_i.y,
		fabsf(origin.z) < ORIGIN ? origin.z + FLOAT_SCALE * n.z : p_i.z
	);
}
