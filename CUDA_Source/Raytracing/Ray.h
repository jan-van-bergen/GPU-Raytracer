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

__device__ float3 ray_origin_epsilon_offset(const float3 & origin, const float3 & direction, const float3 & normal) {
	return origin + sign(dot(direction, normal)) * EPSILON * normal;
}
