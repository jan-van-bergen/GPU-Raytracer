#pragma once
#include <utility>

#include <vector_types.h>
#include <corecrt_math.h>

#include "cuda_math.h"

#include "../Common.h"

struct Ray {
	float3 origin;
	float3 direction;
	float3 direction_inv;
};

struct RayHit {
	float t = INFINITY;
	float u, v;

	int triangle_id;
};

// Triangles in SoA layout
__device__ float3 * triangles_position0;
__device__ float3 * triangles_position_edge1;
__device__ float3 * triangles_position_edge2;

__device__ float3 * triangles_normal0;
__device__ float3 * triangles_normal_edge1;
__device__ float3 * triangles_normal_edge2; 
 
__device__ float2 * triangles_tex_coord0;
__device__ float2 * triangles_tex_coord_edge1;
__device__ float2 * triangles_tex_coord_edge2;

__device__ int * triangles_material_id;

static_assert(MBVH_WIDTH == 4, "The implementation assumes a Quaternary BVH");

struct MBVHNode {
	float4 aabb_min_x;
	float4 aabb_min_y;
	float4 aabb_min_z;
	float4 aabb_max_x;
	float4 aabb_max_y;
	float4 aabb_max_z;
	union {
		int index[MBVH_WIDTH];
		int child[MBVH_WIDTH];
	};
	int count[MBVH_WIDTH];
};

__device__ MBVHNode * mbvh_nodes;

struct AABBHits {
	union {
		float4 t_near;
		float  t_near_f[4];
		int    t_near_i[4];
	};
	bool hit[4];
};

__device__ inline void triangle_trace(int triangle_id, const Ray & ray, RayHit & ray_hit) {
	const float3 & position0      = triangles_position0     [triangle_id];
	const float3 & position_edge1 = triangles_position_edge1[triangle_id];
	const float3 & position_edge2 = triangles_position_edge2[triangle_id];

	float3 h = cross(ray.direction, position_edge2);
	float  a = dot(position_edge1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - position0;
	float  u = f * dot(s, h);

	if (u >= 0.0f && u <= 1.0f) {
		float3 q = cross(s, position_edge1);
		float  v = f * dot(ray.direction, q);

		if (v >= 0.0f && u + v <= 1.0f) {
			float t = f * dot(position_edge2, q);

			if (t > EPSILON && t < ray_hit.t) {
				ray_hit.t = t;
				ray_hit.u = u;
				ray_hit.v = v;
				ray_hit.triangle_id = triangle_id;
			}
		}
	}
}

__device__ inline bool triangle_intersect(int triangle_id, const Ray & ray, float max_distance) {
	const float3 & position0      = triangles_position0     [triangle_id];
	const float3 & position_edge1 = triangles_position_edge1[triangle_id];
	const float3 & position_edge2 = triangles_position_edge2[triangle_id];

	float3 h = cross(ray.direction, position_edge2);
	float  a = dot(position_edge1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - position0;
	float  u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) return false;

	float3 q = cross(s, position_edge1);
	float  v = f * dot(ray.direction, q);

	if (v < 0.0f || u + v > 1.0f) return false;

	float t = f * dot(position_edge2, q);

	if (t < EPSILON || t >= max_distance) return false;

	return true;
}

// Check the Ray agains the four AABB's of the children of the given MBVH Node
__device__ inline AABBHits mbvh_node_intersect(const MBVHNode & node, const Ray & ray, float max_distance) {
	AABBHits result;

	float4 tx0 = (node.aabb_min_x - ray.origin.x) * ray.direction_inv.x;
	float4 tx1 = (node.aabb_max_x - ray.origin.x) * ray.direction_inv.x;
	float4 ty0 = (node.aabb_min_y - ray.origin.y) * ray.direction_inv.y;
	float4 ty1 = (node.aabb_max_y - ray.origin.y) * ray.direction_inv.y;
	float4 tz0 = (node.aabb_min_z - ray.origin.z) * ray.direction_inv.z;
	float4 tz1 = (node.aabb_max_z - ray.origin.z) * ray.direction_inv.z;

	float4 tx_min = fminf(tx0, tx1);
	float4 tx_max = fmaxf(tx0, tx1);
	float4 ty_min = fminf(ty0, ty1);
	float4 ty_max = fmaxf(ty0, ty1);
	float4 tz_min = fminf(tz0, tz1);
	float4 tz_max = fmaxf(tz0, tz1);
	
	result.t_near = fmaxf(fmaxf(make_float4(EPSILON),      tx_min), fmaxf(ty_min, tz_min));
	float4 t_far  = fminf(fminf(make_float4(max_distance), tx_max), fminf(ty_max, tz_max));

	result.hit[0] = result.t_near_f[0] < t_far.x;
	result.hit[1] = result.t_near_f[1] < t_far.y;
	result.hit[2] = result.t_near_f[2] < t_far.z;
	result.hit[3] = result.t_near_f[3] < t_far.w;

	// Use the two least significant bits of the float to store the index
	result.t_near_i[0] = (result.t_near_i[0] & 0xfffffffc) | 0;
	result.t_near_i[1] = (result.t_near_i[1] & 0xfffffffc) | 1;
	result.t_near_i[2] = (result.t_near_i[2] & 0xfffffffc) | 2;
	result.t_near_i[3] = (result.t_near_i[3] & 0xfffffffc) | 3;

	// Bubble sort to order the hit distances
	for (int i = 1; i < MBVH_WIDTH; i++) {
		int j = i - 1;

		while (j >= 0 && result.t_near_f[j] < result.t_near_f[j + 1]) {
			// XOR swap
			result.t_near_i[j    ] ^= result.t_near_i[j + 1];	
			result.t_near_i[j + 1] ^= result.t_near_i[j    ];	
			result.t_near_i[j    ] ^= result.t_near_i[j + 1];	

			j--;
		}
	}

	return result;
}

__device__ inline unsigned pack_mbvh_node(int index, int id) {
	ASSERT(index < 0x3fffffff, "Index must fit in 30 bits");

	return (id << 30) | index;
}

__device__ inline void unpack_mbvh_node(unsigned packed, int & index, int & id) {
	index = packed & 0x3fffffff;
	id    = packed >> 30;
}

__device__ inline void mbvh_trace(const Ray & ray, RayHit & ray_hit) {
	unsigned stack[32];
	int stack_size = 1;

	// Push root on stack
	stack[0] = 1;

	while (stack_size > 0) {
		// Pop Node of the stack
		unsigned packed = stack[--stack_size];

		int node_index, node_id;
		unpack_mbvh_node(packed, node_index, node_id);

		int index = mbvh_nodes[node_index].child[node_id];
		int count = mbvh_nodes[node_index].count[node_id];

		ASSERT(count != -1, "Unpacked invalid Node!");

		// Check if the Node is a leaf
		if (count > 0) {
			for (int j = index; j < index + count; j++) {
				triangle_trace(j, ray, ray_hit);
			}
		} else {
			int child = index;

			AABBHits aabb_hits = mbvh_node_intersect(mbvh_nodes[child], ray, ray_hit.t);
			
			for (int i = 0; i < MBVH_WIDTH; i++) {
				// Extract index from the 2 least significant bits
				int id = aabb_hits.t_near_i[i] & 0b11;
				
				if (aabb_hits.hit[id]) {
					stack[stack_size++] = pack_mbvh_node(child, id);
				}
			}
		}
	}
}

__device__ inline bool mbvh_intersect(const Ray & ray, float max_distance) {
	unsigned stack[32];
	int stack_size = 1;

	// Push root on stack
	stack[0] = 1;

	while (stack_size > 0) {
		// Pop Node of the stack
		unsigned packed = stack[--stack_size];

		int node_index, node_id;
		unpack_mbvh_node(packed, node_index, node_id);

		int index = mbvh_nodes[node_index].index[node_id];
		int count = mbvh_nodes[node_index].count[node_id];

		ASSERT(count != -1, "Unpacked invalid Node!");

		// Check if the Node is a leaf
		if (count > 0) {
			for (int j = index; j < index + count; j++) {
				if (triangle_intersect(j, ray, max_distance)) {
					return true;
				}
			}
		} else {
			int child = index;

			AABBHits aabb_hits = mbvh_node_intersect(mbvh_nodes[child], ray, max_distance);
			
			for (int i = 0; i < MBVH_WIDTH; i++) {
				// Extract index from the 2 least significant bits
				int id = aabb_hits.t_near_i[i] & 0b11;
				
				if (aabb_hits.hit[id]) {
					stack[stack_size++] = pack_mbvh_node(child, id);
				}
			}
		}
	}

	return false;
}
