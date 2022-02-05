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
};

__device__ float3 ray_origin_epsilon_offset(const float3 & origin, const float3 & direction, const float3 & geometric_normal) {
	return origin + sign(dot(direction, geometric_normal)) * EPSILON * geometric_normal;
}
