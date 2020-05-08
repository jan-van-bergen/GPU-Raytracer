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

#if BVH_TYPE == BVH_SBVH
struct AABB {
	float3 min;
	float3 max;

	__device__ inline bool intersects(const Ray & ray, float max_distance) const {
		float3 t0 = (min - ray.origin) * ray.direction_inv;
		float3 t1 = (max - ray.origin) * ray.direction_inv;
		
		float3 t_min = fminf(t0, t1);
		float3 t_max = fmaxf(t0, t1);
		
		float t_near = fmaxf(fmaxf(EPSILON,      t_min.x), fmaxf(t_min.y, t_min.z));
		float t_far  = fminf(fminf(max_distance, t_max.x), fminf(t_max.y, t_max.z));
	
		return t_near < t_far;
	}
};

struct BVHNode {
	AABB aabb;
	union {
		int left;
		int first;
	};
	int count;

	__device__ inline bool is_leaf() const {
		return (count & (~BVH_AXIS_MASK)) > 0;
	}

	__device__ inline bool should_visit_left_first(const Ray & ray) const {
		switch (count & BVH_AXIS_MASK) {
			case BVH_AXIS_X_BITS: return ray.direction.x > 0.0f;
			case BVH_AXIS_Y_BITS: return ray.direction.y > 0.0f;
			case BVH_AXIS_Z_BITS: return ray.direction.z > 0.0f;
		}
	}
};

__device__ BVHNode * bvh_nodes;

__device__ void bvh_trace(const Ray & ray, RayHit & ray_hit) {
	int stack[BVH_STACK_SIZE];
	int stack_size = 1;

	// Push root on stack
	stack[0] = 0;

	while (stack_size > 0) {
		// Pop Node of the stack
		const BVHNode & node = bvh_nodes[stack[--stack_size]];

		if (node.aabb.intersects(ray, ray_hit.t)) {
			if (node.is_leaf()) {
				for (int i = node.first; i < node.first + node.count; i++) {
					triangle_trace(i, ray, ray_hit);
				}
			} else {
				if (node.should_visit_left_first(ray)) {
					stack[stack_size++] = node.left + 1;
					stack[stack_size++] = node.left;
				} else {
					stack[stack_size++] = node.left;
					stack[stack_size++] = node.left + 1;
				}
			}
		}
	}
}

__device__ bool bvh_intersect(const Ray & ray, float max_distance) {
	int stack[BVH_STACK_SIZE];
	int stack_size = 1;

	// Push root on stack
	stack[0] = 0;

	while (stack_size > 0) {
		// Pop Node of the stack
		const BVHNode & node = bvh_nodes[stack[--stack_size]];

		if (node.aabb.intersects(ray, max_distance)) {
			if (node.is_leaf()) {
				for (int i = node.first; i < node.first + node.count; i++) {
					if (triangle_intersect(i, ray, max_distance)) {
						return true;
					}
				}
			} else {
				if (node.should_visit_left_first(ray)) {
					stack[stack_size++] = node.left + 1;
					stack[stack_size++] = node.left;
				} else {
					stack[stack_size++] = node.left;
					stack[stack_size++] = node.left + 1;
				}
			}
		}
	}

	return false;
}
#elif BVH_TYPE == BVH_QBVH

struct QBVHNode {
	float4 aabb_min_x;
	float4 aabb_min_y;
	float4 aabb_min_z;
	float4 aabb_max_x;
	float4 aabb_max_y;
	float4 aabb_max_z;
	int2 index_and_count[4];
};

__device__ QBVHNode * qbvh_nodes;

struct AABBHits {
	union {
		float4 t_near;
		float  t_near_f[4];
		int    t_near_i[4];
	};
	bool hit[4];
};

// Check the Ray agains the four AABB's of the children of the given QBVH Node
__device__ inline AABBHits qbvh_node_intersect(const QBVHNode & node, const Ray & ray, float max_distance) {
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
	#pragma unroll
	for (int i = 1; i < 4; i++) {
		#pragma unroll
		for (int j = i - 1; j >= 0; j--) {
			if (result.t_near_f[j] < result.t_near_f[j + 1]) {
				// int temp             = result.t_near_i[j  ];
				// result.t_near_i[j  ] = result.t_near_i[j+1];
				// result.t_near_i[j+1] = temp;
				result.t_near_i[j    ] ^= result.t_near_i[j + 1];	
				result.t_near_i[j + 1] ^= result.t_near_i[j    ];	
				result.t_near_i[j    ] ^= result.t_near_i[j + 1];	
			}
		}
	}

	return result;
}

__device__ inline unsigned pack_qbvh_node(int index, int id) {
	ASSERT(index < 0x3fffffff, "Index must fit in 30 bits");

	return (id << 30) | index;
}

__device__ inline void unpack_qbvh_node(unsigned packed, int & index, int & id) {
	index = packed & 0x3fffffff;
	id    = packed >> 30;
}

__device__ inline void bvh_trace(const Ray & ray, RayHit & ray_hit) {
	unsigned stack[BVH_STACK_SIZE];
	int stack_size = 1;

	// Push root on stack
	stack[0] = 1;

	while (stack_size > 0) {
		assert(stack_size <= BVH_STACK_SIZE);

		// Pop Node of the stack
		unsigned packed = stack[--stack_size];

		int node_index, node_id;
		unpack_qbvh_node(packed, node_index, node_id);

		int2 index_and_count = qbvh_nodes[node_index].index_and_count[node_id];

		int index = index_and_count.x;
		int count = index_and_count.y;

		ASSERT(index != -1 && count != -1, "Unpacked invalid Node!");

		// Check if the Node is a leaf
		if (count > 0) {
			for (int j = index; j < index + count; j++) {
				triangle_trace(j, ray, ray_hit);
			}
		} else {
			int child = index;

			AABBHits aabb_hits = qbvh_node_intersect(qbvh_nodes[child], ray, ray_hit.t);
			
			for (int i = 0; i < 4; i++) {
				// Extract index from the 2 least significant bits
				int id = aabb_hits.t_near_i[i] & 0b11;
				
				if (aabb_hits.hit[id]) {
					stack[stack_size++] = pack_qbvh_node(child, id);
				}
			}
		}
	}
}

__device__ inline bool bvh_intersect(const Ray & ray, float max_distance) {
	unsigned stack[BVH_STACK_SIZE];
	int stack_size = 1;

	// Push root on stack
	stack[0] = 1;

	while (stack_size > 0) {
		// Pop Node of the stack
		unsigned packed = stack[--stack_size];

		int node_index, node_id;
		unpack_qbvh_node(packed, node_index, node_id);

		int2 index_and_count = qbvh_nodes[node_index].index_and_count[node_id];

		int index = index_and_count.x;
		int count = index_and_count.y;

		ASSERT(index != -1 && count != -1, "Unpacked invalid Node!");

		// Check if the Node is a leaf
		if (count > 0) {
			for (int j = index; j < index + count; j++) {
				if (triangle_intersect(j, ray, max_distance)) {
					return true;
				}
			}
		} else {
			int child = index;

			AABBHits aabb_hits = qbvh_node_intersect(qbvh_nodes[child], ray, max_distance);
			
			for (int i = 0; i < 4; i++) {
				// Extract index from the 2 least significant bits
				int id = aabb_hits.t_near_i[i] & 0b11;
				
				if (aabb_hits.hit[id]) {
					stack[stack_size++] = pack_qbvh_node(child, id);
				}
			}
		}
	}

	return false;
}
#elif BVH_TYPE == BVH_CWBVH
typedef unsigned char byte;

struct CWBVH {
	float4 node_0;
	float4 node_1;
	float4 node_2;
	float4 node_3;
	float4 node_4; // Node is stored as 5 float4's so we can load the entire 80 bytes in 5 global memory accesses
};

__device__ CWBVH * cwbvh_nodes;

__device__ __inline__ inline unsigned cwbvh_node_intersect(
	const Ray & ray,
	unsigned oct_inv4,
	bool ray_negative_x,
	bool ray_negative_y,
	bool ray_negative_z,
	float max_distance,
	const float4 & node_0, const float4 & node_1, const float4 & node_2, const float4 & node_3, const float4 & node_4) {

	float3 p = make_float3(node_0);

	unsigned e_imask = float_as_uint(node_0.w);
	byte e_x   = extract_byte(e_imask, 0);
	byte e_y   = extract_byte(e_imask, 1);
	byte e_z   = extract_byte(e_imask, 2);
	
	float3 adjusted_ray_direction_inv = make_float3(
		uint_as_float(e_x << 23) * ray.direction_inv.x,
		uint_as_float(e_y << 23) * ray.direction_inv.y,
		uint_as_float(e_z << 23) * ray.direction_inv.z
	);
	float3 adjusted_ray_origin = (p - ray.origin) * ray.direction_inv;

	unsigned hit_mask = 0;

	#pragma unroll
	for (int i = 0; i < 2; i++) {
		unsigned meta4 = float_as_uint(i == 0 ? node_1.z : node_1.w);

		unsigned is_inner4   = (meta4 & (meta4 << 1)) & 0x10101010;
		unsigned inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
		unsigned bit_index4  = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1f1f1f1f;
		unsigned child_bits4 = (meta4 >> 5) & 0x07070707;

		// @SPEED: use PRMT

		// Select near and far planes based on ray octant
		unsigned q_lo_x = ray_negative_x ? float_as_uint(i == 0 ? node_2.z : node_2.w) : float_as_uint(i == 0 ? node_2.x : node_2.y);
		unsigned q_hi_x = ray_negative_x ? float_as_uint(i == 0 ? node_2.x : node_2.y) : float_as_uint(i == 0 ? node_2.z : node_2.w);

		unsigned q_lo_y = ray_negative_y ? float_as_uint(i == 0 ? node_3.z : node_3.w) : float_as_uint(i == 0 ? node_3.x : node_3.y);
		unsigned q_hi_y = ray_negative_y ? float_as_uint(i == 0 ? node_3.x : node_3.y) : float_as_uint(i == 0 ? node_3.z : node_3.w);

		unsigned q_lo_z = ray_negative_z ? float_as_uint(i == 0 ? node_4.z : node_4.w) : float_as_uint(i == 0 ? node_4.x : node_4.y);
		unsigned q_hi_z = ray_negative_z ? float_as_uint(i == 0 ? node_4.x : node_4.y) : float_as_uint(i == 0 ? node_4.z : node_4.w);

		#pragma unroll
		for (int j = 0; j < 4; j++) {
			// Extract j-th byte
			float3 tmin = make_float3(float(extract_byte(q_lo_x, j)), float(extract_byte(q_lo_y, j)), float(extract_byte(q_lo_z, j)));
			float3 tmax = make_float3(float(extract_byte(q_hi_x, j)), float(extract_byte(q_hi_y, j)), float(extract_byte(q_hi_z, j)));

			// Account for grid origin and scale
			tmin = tmin * adjusted_ray_direction_inv + adjusted_ray_origin;
			tmax = tmax * adjusted_ray_direction_inv + adjusted_ray_origin;

			// @TODO: VMIN, VMAX
			float t_lo = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.z, EPSILON));
			float t_hi = fminf(fminf(tmax.x, tmax.y), fminf(tmax.z, max_distance));

			bool intersected = t_lo < t_hi;
			if (intersected) {
				unsigned child_bits = extract_byte(child_bits4, j);
				unsigned bit_index  = extract_byte(bit_index4,  j);

				hit_mask |= child_bits << bit_index;
			}
		}
	}

	return hit_mask;
}

__device__ inline void bvh_trace(const Ray & ray, RayHit & ray_hit) {
	bool ray_negative_x = ray.direction.x < 0.0f;
	bool ray_negative_y = ray.direction.y < 0.0f;
	bool ray_negative_z = ray.direction.z < 0.0f;

	unsigned oct = 
		(ray_negative_x < 0.0f ? 0b100 : 0) |
		(ray_negative_y < 0.0f ? 0b010 : 0) |
		(ray_negative_z < 0.0f ? 0b001 : 0);

	unsigned oct_inv  = 7 - oct;
	unsigned oct_inv4 = oct_inv * 0x01010101;

	uint2 stack[BVH_STACK_SIZE];
	int  stack_size = 0;

	uint2 current_group = make_uint2(0, 0x80000000);

	while (true) {
		uint2 triangle_group;

		if (current_group.y > 0x00ffffff) {
			unsigned hits_imask = current_group.y;

			unsigned child_index_offset = msb(hits_imask);
			unsigned child_index_base   = current_group.x;

			// Remove n from current_group;
			current_group.y &= ~(1 << child_index_offset);

			// If the node group is not yet empty, push it on the stack
			if (current_group.y > 0x00ffffff) {
				assert(stack_size < BVH_STACK_SIZE);

				stack[stack_size++] = current_group;
			}

			unsigned slot_index     = (child_index_offset - 24) ^ oct_inv;
			unsigned relative_index = __popc(hits_imask & ~(0xffffffff << slot_index));

			unsigned child_node_index = child_index_base + relative_index;

			float4 node_0 = cwbvh_nodes[child_node_index].node_0;
			float4 node_1 = cwbvh_nodes[child_node_index].node_1;
			float4 node_2 = cwbvh_nodes[child_node_index].node_2;
			float4 node_3 = cwbvh_nodes[child_node_index].node_3;
			float4 node_4 = cwbvh_nodes[child_node_index].node_4;

			unsigned hitmask = cwbvh_node_intersect(ray, oct_inv4, ray_negative_x, ray_negative_y, ray_negative_z, ray_hit.t, node_0, node_1, node_2, node_3, node_4);

			byte imask = extract_byte(float_as_uint(node_0.w), 3);
			
			current_group .x = float_as_uint(node_1.x); // Child    base offset
			triangle_group.x = float_as_uint(node_1.y); // Triangle base offset

			current_group .y = (hitmask & 0xff000000) | unsigned(imask);
			triangle_group.y = (hitmask & 0x00ffffff);
		} else {
			triangle_group = current_group;
			current_group  = make_uint2(0);
		}

		int active_threads = __popc(__activemask());

		// While the triangle group is not empty
		while (triangle_group.y != 0) {
			// if (__popc(__activemask()) < active_threads / 4) {
			// 	stack[stack_size++] = triangle_group;

			// 	break;
			// }

			int triangle_index = msb(triangle_group.y);

			triangle_group.y &= ~(1 << triangle_index);

			triangle_trace(triangle_group.x + triangle_index, ray, ray_hit);
		}

		if (current_group.y <= 0x00ffffff) {
			if (stack_size == 0) break;

			current_group = stack[--stack_size];
		}
	}
}

__device__ inline bool bvh_intersect(const Ray & ray, float max_distance) {
	bool ray_negative_x = ray.direction.x < 0.0f;
	bool ray_negative_y = ray.direction.y < 0.0f;
	bool ray_negative_z = ray.direction.z < 0.0f;

	unsigned oct = 
		(ray_negative_x < 0.0f ? 0b100 : 0) |
		(ray_negative_y < 0.0f ? 0b010 : 0) |
		(ray_negative_z < 0.0f ? 0b001 : 0);

	unsigned oct_inv  = 7 - oct;
	unsigned oct_inv4 = oct_inv * 0x01010101;

	uint2 stack[BVH_STACK_SIZE];
	int  stack_size = 0;

	uint2 current_group = make_uint2(0, 0x80000000);

	while (true) {
		uint2 triangle_group;

		if (current_group.y > 0x00ffffff) {
			unsigned hits_imask = current_group.y;

			unsigned child_index_offset = msb(hits_imask);
			unsigned child_index_base   = current_group.x;

			// Remove n from current_group;
			current_group.y &= ~(1 << child_index_offset);

			// If the node group is not yet empty, push it on the stack
			if (current_group.y > 0x00ffffff) {
				assert(stack_size < BVH_STACK_SIZE);

				stack[stack_size++] = current_group;
			}

			unsigned slot_index     = (child_index_offset - 24) ^ oct_inv;
			unsigned relative_index = __popc(hits_imask & ~(0xffffffff << slot_index));

			unsigned child_node_index = child_index_base + relative_index;

			float4 node_0 = cwbvh_nodes[child_node_index].node_0;
			float4 node_1 = cwbvh_nodes[child_node_index].node_1;
			float4 node_2 = cwbvh_nodes[child_node_index].node_2;
			float4 node_3 = cwbvh_nodes[child_node_index].node_3;
			float4 node_4 = cwbvh_nodes[child_node_index].node_4;

			unsigned hitmask = cwbvh_node_intersect(ray, oct_inv4, ray_negative_x, ray_negative_y, ray_negative_z, max_distance, node_0, node_1, node_2, node_3, node_4);

			byte imask = extract_byte(float_as_uint(node_0.w), 3);
			
			current_group .x = float_as_uint(node_1.x); // Child    base offset
			triangle_group.x = float_as_uint(node_1.y); // Triangle base offset

			current_group .y = (hitmask & 0xff000000) | unsigned(imask);
			triangle_group.y = (hitmask & 0x00ffffff);
		} else {
			triangle_group = current_group;
			current_group  = make_uint2(0);
		}

		int active_threads = __popc(__activemask());

		// While the triangle group is not empty
		while (triangle_group.y != 0) {
			// if (__popc(__activemask()) < active_threads / 4) {
			// 	stack[stack_size++] = triangle_group;

			// 	break;
			// }

			int triangle_index = msb(triangle_group.y);

			triangle_group.y &= ~(1 << triangle_index);

			if (triangle_intersect(triangle_group.x + triangle_index, ray, max_distance)) {
				return true;
			}
		}

		if (current_group.y <= 0x00ffffff) {
			if (stack_size == 0) break;

			current_group = stack[--stack_size];
		}
	}

	return false;
}
#endif
