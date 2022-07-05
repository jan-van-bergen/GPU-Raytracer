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

__device__ float3 reflect_direction(const float3 & direction, const float3 & normal) {
	return 2.0f * dot(direction, normal) * normal - direction;
}

__device__ float3 refract_direction(const float3 & direction, const float3 & normal, float eta) {
	float cos_theta = dot(direction, normal);
	float k = 1.0f - eta*eta * (1.0f - square(cos_theta));
	return (eta * cos_theta - safe_sqrt(k)) * normal - eta * direction;
}
