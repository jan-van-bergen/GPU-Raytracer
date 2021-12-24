#pragma once
#include "Ray.h"

struct Triangle {
	float4 part_0; // position_0       xyz and position_edge_1  x
	float4 part_1; // position_edge_1   yz and position_edge_2  xy
	float4 part_2; // position_edge_2    z and normal_0         xyz
	float4 part_3; // normal_edge_1    xyz and normal_edge_2    x
	float4 part_4; // normal_edge_2     yz and tex_coord_0      xy
	float4 part_5; // tex_coord_edge_1 xy  and tex_coord_edge_2 xy
};

__device__ __constant__ const Triangle * triangles;

struct TrianglePos {
	float3 position_0;
	float3 position_edge_1;
	float3 position_edge_2;
};

__device__ inline TrianglePos triangle_get_positions(int index) {
	float4 part_0 = __ldg(&triangles[index].part_0);
	float4 part_1 = __ldg(&triangles[index].part_1);
	float4 part_2 = __ldg(&triangles[index].part_2);

	TrianglePos triangle;

	triangle.position_0      = make_float3(part_0.x, part_0.y, part_0.z);
	triangle.position_edge_1 = make_float3(part_0.w, part_1.x, part_1.y);
	triangle.position_edge_2 = make_float3(part_1.z, part_1.w, part_2.x);

	return triangle;
}

struct TrianglePosNor {
	float3 position_0;
	float3 position_edge_1;
	float3 position_edge_2;

	float3 normal_0;
	float3 normal_edge_1;
	float3 normal_edge_2;
};

__device__ inline TrianglePosNor triangle_get_positions_and_normals(int index) {
	float4 part_0 = __ldg(&triangles[index].part_0);
	float4 part_1 = __ldg(&triangles[index].part_1);
	float4 part_2 = __ldg(&triangles[index].part_2);
	float4 part_3 = __ldg(&triangles[index].part_3);
	float4 part_4 = __ldg(&triangles[index].part_4);

	TrianglePosNor triangle;

	triangle.position_0      = make_float3(part_0.x, part_0.y, part_0.z);
	triangle.position_edge_1 = make_float3(part_0.w, part_1.x, part_1.y);
	triangle.position_edge_2 = make_float3(part_1.z, part_1.w, part_2.x);

	triangle.normal_0      = make_float3(part_2.y, part_2.z, part_2.w);
	triangle.normal_edge_1 = make_float3(part_3.x, part_3.y, part_3.z);
	triangle.normal_edge_2 = make_float3(part_3.w, part_4.x, part_4.y);

	return triangle;
};

struct TrianglePosNorTex {
	float3 position_0;
	float3 position_edge_1;
	float3 position_edge_2;

	float3 normal_0;
	float3 normal_edge_1;
	float3 normal_edge_2;

	float2 tex_coord_0;
	float2 tex_coord_edge_1;
	float2 tex_coord_edge_2;
};

__device__ inline TrianglePosNorTex triangle_get_positions_normals_and_tex_coords(int index) {
	float4 part_0 = __ldg(&triangles[index].part_0);
	float4 part_1 = __ldg(&triangles[index].part_1);
	float4 part_2 = __ldg(&triangles[index].part_2);
	float4 part_3 = __ldg(&triangles[index].part_3);
	float4 part_4 = __ldg(&triangles[index].part_4);
	float4 part_5 = __ldg(&triangles[index].part_5);

	TrianglePosNorTex triangle;

	triangle.position_0      = make_float3(part_0.x, part_0.y, part_0.z);
	triangle.position_edge_1 = make_float3(part_0.w, part_1.x, part_1.y);
	triangle.position_edge_2 = make_float3(part_1.z, part_1.w, part_2.x);

	triangle.normal_0      = make_float3(part_2.y, part_2.z, part_2.w);
	triangle.normal_edge_1 = make_float3(part_3.x, part_3.y, part_3.z);
	triangle.normal_edge_2 = make_float3(part_3.w, part_4.x, part_4.y);

	triangle.tex_coord_0      = make_float2(part_4.z, part_4.w);
	triangle.tex_coord_edge_1 = make_float2(part_5.x, part_5.y);
	triangle.tex_coord_edge_2 = make_float2(part_5.z, part_5.w);

	return triangle;
}

// Triangle texture base LOD as described in "Texture Level of Detail Strategies for Real-Time Ray Tracing"
__device__ inline float triangle_get_lod(
	float          mesh_scale,
	float          triangle_area_inv,
	const float2 & tex_coord_edge_1,
	const float2 & tex_coord_edge_2
) {
	float t_a = fabsf(
		tex_coord_edge_1.x * tex_coord_edge_2.y -
		tex_coord_edge_2.x * tex_coord_edge_1.y
	);

	return t_a * triangle_area_inv / (mesh_scale * mesh_scale);
}

__device__ inline float triangle_get_curvature(
	const float3 & position_edge_1,
	const float3 & position_edge_2,
	const float3 & normal_edge_1,
	const float3 & normal_edge_2
) {
	float3 normal_edge_0   = normal_edge_1   - normal_edge_2;
	float3 position_edge_0 = position_edge_1 - position_edge_2;

	float k_01 = dot(normal_edge_1, position_edge_1) / dot(position_edge_1, position_edge_1);
	float k_02 = dot(normal_edge_2, position_edge_2) / dot(position_edge_2, position_edge_2);
	float k_12 = dot(normal_edge_0, position_edge_0) / dot(position_edge_0, position_edge_0);

	return (k_01 + k_02 + k_12) * (1.0f / 3.0f); // Eq. 6 (Akenine-Möller 2021)
}

__device__ inline void triangle_barycentric(const TrianglePosNor & triangle, float u, float v, float3 & position, float3 & normal) {
	position = barycentric(u, v, triangle.position_0,  triangle.position_edge_1,  triangle.position_edge_2);
	normal   = barycentric(u, v, triangle.normal_0,    triangle.normal_edge_1,    triangle.normal_edge_2);
}

__device__ inline void triangle_barycentric(const TrianglePosNorTex & triangle, float u, float v, float3 & position, float3 & normal, float2 & tex_coord) {
	position  = barycentric(u, v, triangle.position_0,  triangle.position_edge_1,  triangle.position_edge_2);
	normal    = barycentric(u, v, triangle.normal_0,    triangle.normal_edge_1,    triangle.normal_edge_2);
	tex_coord = barycentric(u, v, triangle.tex_coord_0, triangle.tex_coord_edge_1, triangle.tex_coord_edge_2);
}

__device__ inline void triangle_intersect(int mesh_id, int triangle_id, const Ray & ray, RayHit & ray_hit) {
	TrianglePos triangle = triangle_get_positions(triangle_id);

	float3 h = cross(ray.direction, triangle.position_edge_2);
	float  a = dot(triangle.position_edge_1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - triangle.position_0;
	float  u = f * dot(s, h);

	if (u >= 0.0f && u <= 1.0f) {
		float3 q = cross(s, triangle.position_edge_1);
		float  v = f * dot(ray.direction, q);

		if (v >= 0.0f && u + v <= 1.0f) {
			float t = f * dot(triangle.position_edge_2, q);

			if (t > 0.0f && t < ray_hit.t) {
				ray_hit.t = t;
				ray_hit.u = u;
				ray_hit.v = v;
				ray_hit.mesh_id     = mesh_id;
				ray_hit.triangle_id = triangle_id;
			}
		}
	}
}

__device__ inline bool triangle_intersect_shadow(int triangle_id, const Ray & ray, float max_distance) {
	TrianglePos triangle = triangle_get_positions(triangle_id);

	float3 h = cross(ray.direction, triangle.position_edge_2);
	float  a = dot(triangle.position_edge_1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - triangle.position_0;
	float  u = f * dot(s, h);

	if (u >= 0.0f && u <= 1.0f) {
		float3 q = cross(s, triangle.position_edge_1);
		float  v = f * dot(ray.direction, q);

		if (v >= 0.0f && u + v <= 1.0f) {
			float t = f * dot(triangle.position_edge_2, q);

			if (t > 0.0f && t < max_distance) return true;
		}
	}

	return false;
}
