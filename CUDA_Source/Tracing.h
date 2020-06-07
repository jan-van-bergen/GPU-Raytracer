#pragma once
#include <utility>

#include <vector_types.h>
#include <corecrt_math.h>

#include "cuda_math.h"

#include "../Common.h"

#define TRIANGLE_POSTPONING false

struct Ray {
	float3 origin;
	float3 direction;
	float3 direction_inv;
};

struct RayHit {
	float t = INFINITY;
	float u, v;

	int triangle_id = -1;
};

struct Triangle {
	float4 part_0; // position_0       xyz and position_edge_1  x
	float4 part_1; // position_edge_1   yz and position_edge_2  xy
	float4 part_2; // position_edge_2    z and normal_0         xyz
	float4 part_3; // normal_edge_1    xyz and normal_edge_2    x
	float4 part_4; // normal_edge_2     yz and tex_coord_0      xy
	float4 part_5; // tex_coord_edge_1 xy  and tex_coord_edge_2 xy
};

__device__ Triangle * triangles;
__device__ int      * triangle_material_ids;

__device__ inline int triangle_get_material_id(int index) {
	return triangle_material_ids[index];
}

__device__ inline void triangle_get_positions(int index, float3 & position_0, float3 & position_edge_1, float3 & position_edge_2) {
	float4 part_0 = triangles[index].part_0;
	float4 part_1 = triangles[index].part_1;
	float4 part_2 = triangles[index].part_2;

	position_0      = make_float3(part_0.x, part_0.y, part_0.z);
	position_edge_1 = make_float3(part_0.w, part_1.x, part_1.y);
	position_edge_2 = make_float3(part_1.z, part_1.w, part_2.x);
}

__device__ inline void triangle_get_positions_and_normals(int index, 
	float3 & position_0, float3 & position_edge_1, float3 & position_edge_2,
	float3 & normal_0,   float3 & normal_edge_1,   float3 & normal_edge_2
) {
	float4 part_0 = triangles[index].part_0;
	float4 part_1 = triangles[index].part_1;
	float4 part_2 = triangles[index].part_2;
	float4 part_3 = triangles[index].part_3;
	float4 part_4 = triangles[index].part_4;

	position_0      = make_float3(part_0.x, part_0.y, part_0.z);
	position_edge_1 = make_float3(part_0.w, part_1.x, part_1.y);
	position_edge_2 = make_float3(part_1.z, part_1.w, part_2.x);

	normal_0      = make_float3(part_2.y, part_2.z, part_3.w);
	normal_edge_1 = make_float3(part_3.x, part_3.y, part_3.z);
	normal_edge_2 = make_float3(part_3.w, part_4.x, part_4.y);
}

__device__ inline void triangle_get_positions_normals_and_tex_coords(int index, 
	float3 & position_0,  float3 & position_edge_1,  float3 & position_edge_2,
	float3 & normal_0,    float3 & normal_edge_1,    float3 & normal_edge_2,
	float2 & tex_coord_0, float2 & tex_coord_edge_1, float2 & tex_coord_edge_2
) {
	float4 part_0 = triangles[index].part_0;
	float4 part_1 = triangles[index].part_1;
	float4 part_2 = triangles[index].part_2;
	float4 part_3 = triangles[index].part_3;
	float4 part_4 = triangles[index].part_4;
	float4 part_5 = triangles[index].part_5;

	position_0      = make_float3(part_0.x, part_0.y, part_0.z);
	position_edge_1 = make_float3(part_0.w, part_1.x, part_1.y);
	position_edge_2 = make_float3(part_1.z, part_1.w, part_2.x);

	normal_0      = make_float3(part_2.y, part_2.z, part_2.w);
	normal_edge_1 = make_float3(part_3.x, part_3.y, part_3.z);
	normal_edge_2 = make_float3(part_3.w, part_4.x, part_4.y);

	tex_coord_0      = make_float2(part_4.z, part_4.w);
	tex_coord_edge_1 = make_float2(part_5.x, part_5.y);
	tex_coord_edge_2 = make_float2(part_5.z, part_5.w);
}

__device__ inline void triangle_trace(int triangle_id, const Ray & ray, RayHit & ray_hit) {
	float3 position_0;
	float3 position_edge_1;
	float3 position_edge_2;
	triangle_get_positions(triangle_id, position_0, position_edge_1, position_edge_2);

	float3 h = cross(ray.direction, position_edge_2);
	float  a = dot(position_edge_1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - position_0;
	float  u = f * dot(s, h);

	if (u >= 0.0f && u <= 1.0f) {
		float3 q = cross(s, position_edge_1);
		float  v = f * dot(ray.direction, q);

		if (v >= 0.0f && u + v <= 1.0f) {
			float t = f * dot(position_edge_2, q);

			if (t > EPSILON && t < ray_hit.t) {
				ray_hit.t = t;
				ray_hit.u = u;
				ray_hit.v = v;
				ray_hit.triangle_id = triangle_id;
			}
		}
	}
}

__device__ inline bool triangle_trace_shadow(int triangle_id, const Ray & ray, float max_distance) {
	float3 position_0;
	float3 position_edge_1;
	float3 position_edge_2;
	triangle_get_positions(triangle_id, position_0, position_edge_1, position_edge_2);

	float3 h = cross(ray.direction, position_edge_2);
	float  a = dot(position_edge_1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - position_0;
	float  u = f * dot(s, h);

	if (u >= 0.0f && u <= 1.0f) {
		float3 q = cross(s, position_edge_1);
		float  v = f * dot(ray.direction, q);

		if (v >= 0.0f && u + v <= 1.0f) {
			float t = f * dot(position_edge_2, q);

			if (t > EPSILON && t < max_distance) return true;
		}
	}

	return false;
}

// Function that decides whether to push on the shared stack or thread local stack
template<typename T, int N>
__device__ __inline__ inline void stack_push(T shared_stack[WARP_SIZE][N][SHARED_STACK_SIZE], T stack[BVH_STACK_SIZE - N], int & stack_size, T item) {
	// assert(stack_size < BVH_STACK_SIZE);

	if (stack_size < SHARED_STACK_SIZE) {
		shared_stack[threadIdx.x][threadIdx.y][stack_size] = item;
	} else {
		stack[stack_size - SHARED_STACK_SIZE] = item;
	}
	stack_size++;
}

// Function that decides whether to pop from the shared stack or thread local stack
template<typename T, int N>
__device__ __inline__ inline T stack_pop(const T shared_stack[WARP_SIZE][N][SHARED_STACK_SIZE], const T stack[BVH_STACK_SIZE - N], int & stack_size) {
	// assert(stack_size > 0);

	stack_size--;
	if (stack_size < SHARED_STACK_SIZE) {
		return shared_stack[threadIdx.x][threadIdx.y][stack_size];
	} else {
		return stack[stack_size - SHARED_STACK_SIZE];
	}
}

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
struct AABB {
	float3 min;
	float3 max;

	__device__ inline bool intersects(const Ray & ray, float max_distance) const {
		float3 t0 = (min - ray.origin) * ray.direction_inv;
		float3 t1 = (max - ray.origin) * ray.direction_inv;

		float t_near = vmin_max(t0.x, t1.x, vmin_max(t0.y, t1.y, vmin_max(t0.z, t1.z, EPSILON)));
		float t_far  = vmax_min(t0.x, t1.x, vmax_min(t0.y, t1.y, vmax_min(t0.z, t1.z, max_distance)));
	
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

__device__ void bvh_trace(int ray_count, int * rays_retired) {
	__shared__ int shared_stack[WARP_SIZE][TRACE_BLOCK_Y][SHARED_STACK_SIZE];

	int stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int stack_size = 0;

	int    ray_index;
	Ray    ray;
	RayHit ray_hit;

	while (true) {
		bool inactive = stack_size == 0;

		if (inactive) {
			ray_index = atomic_agg_inc(rays_retired);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_trace.origin   .to_float3(ray_index);
			ray.direction = ray_buffer_trace.direction.to_float3(ray_index);
			ray.direction_inv = make_float3(
				1.0f / ray.direction.x, 
				1.0f / ray.direction.y, 
				1.0f / ray.direction.z
			);

			ray_hit.t           = INFINITY;
			ray_hit.triangle_id = -1;

			// Push root on stack
			stack_size                                = 1;
			shared_stack[threadIdx.x][threadIdx.y][0] = 0;
		}

		while (true) {
			// Pop Node of the stack
			int child = stack_pop(shared_stack, stack, stack_size);

			const BVHNode & node = bvh_nodes[child];

			if (node.aabb.intersects(ray, ray_hit.t)) {
				if (node.is_leaf()) {
					for (int i = node.first; i < node.first + node.count; i++) {
						triangle_trace(i, ray, ray_hit);
					}
				} else {
					int first, second;

					if (node.should_visit_left_first(ray)) {
						second = node.left + 1;
						first  = node.left;
					} else {
						second = node.left;
						first = node.left + 1;
					}

					stack_push(shared_stack, stack, stack_size, second);
					stack_push(shared_stack, stack, stack_size, first);
				}
			}

			if (stack_size == 0) {
				ray_buffer_trace.triangle_id[ray_index] = ray_hit.triangle_id;
				ray_buffer_trace.u[ray_index] = ray_hit.u;
				ray_buffer_trace.v[ray_index] = ray_hit.v;

				break;
			}
		}
	}
}

__device__ void bvh_trace_shadow(int ray_count, int * rays_retired, int bounce) {
	__shared__ int shared_stack[WARP_SIZE][SHADOW_TRACE_BLOCK_Y][SHARED_STACK_SIZE];

	int stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int stack_size = 0;

	int ray_index;
	Ray ray;
	
	float max_distance;

	while (true) {
		bool inactive = stack_size == 0;

		if (inactive) {
			ray_index = atomic_agg_inc(rays_retired);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_shadow.ray_origin   .to_float3(ray_index);
			ray.direction = ray_buffer_shadow.ray_direction.to_float3(ray_index);
			ray.direction_inv = make_float3(
				1.0f / ray.direction.x, 
				1.0f / ray.direction.y, 
				1.0f / ray.direction.z
			);

			max_distance = ray_buffer_shadow.max_distance[ray_index];

			// Push root on stack
			stack_size                                = 1;
			shared_stack[threadIdx.x][threadIdx.y][0] = 0;
		}

		while (true) {
			// Pop Node of the stack
			int child = stack_pop(shared_stack, stack, stack_size);

			const BVHNode & node = bvh_nodes[child];

			if (node.aabb.intersects(ray, max_distance)) {
				if (node.is_leaf()) {
					bool hit = false;

					for (int i = node.first; i < node.first + node.count; i++) {
						if (triangle_trace_shadow(i, ray, max_distance)) {
							hit = true;

							break;
						}
					}
					
					if (hit) {
						stack_size = 0;

						break;
					}
				} else {
					int first, second;

					if (node.should_visit_left_first(ray)) {
						second = node.left + 1;
						first  = node.left;
					} else {
						second = node.left;
						first = node.left + 1;
					}

					stack_push(shared_stack, stack, stack_size, second);
					stack_push(shared_stack, stack, stack_size, first);
				}
			}

			if (stack_size == 0) {
				// We didn't hit anything, apply illumination
				int    pixel_index  = ray_buffer_shadow.pixel_index[ray_index];
				float3 illumination = ray_buffer_shadow.illumination.to_float3(ray_index);

				if (bounce == 0) {
					frame_buffer_direct[pixel_index] += make_float4(illumination);
				} else {
					frame_buffer_indirect[pixel_index] += make_float4(illumination);
				}

				break;
			}
		}
	}
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

	result.t_near = make_float4(
		vmin_max(tx0.x, tx1.x, vmin_max(ty0.x, ty1.x, vmin_max(tz0.x, tz1.x, EPSILON))),
		vmin_max(tx0.y, tx1.y, vmin_max(ty0.y, ty1.y, vmin_max(tz0.y, tz1.y, EPSILON))),
		vmin_max(tx0.z, tx1.z, vmin_max(ty0.z, ty1.z, vmin_max(tz0.z, tz1.z, EPSILON))),
		vmin_max(tx0.w, tx1.w, vmin_max(ty0.w, ty1.w, vmin_max(tz0.w, tz1.w, EPSILON)))
	); 
	float4 t_far = make_float4(
		vmax_min(tx0.x, tx1.x, vmax_min(ty0.x, ty1.x, vmax_min(tz0.x, tz1.x, max_distance))),
		vmax_min(tx0.y, tx1.y, vmax_min(ty0.y, ty1.y, vmax_min(tz0.y, tz1.y, max_distance))),
		vmax_min(tx0.z, tx1.z, vmax_min(ty0.z, ty1.z, vmax_min(tz0.z, tz1.z, max_distance))),
		vmax_min(tx0.w, tx1.w, vmax_min(ty0.w, ty1.w, vmax_min(tz0.w, tz1.w, max_distance)))
	);

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

__device__ inline void bvh_trace(int ray_count, int * rays_retired) {
	__shared__ unsigned shared_stack[WARP_SIZE][TRACE_BLOCK_Y][SHARED_STACK_SIZE];

	unsigned stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int stack_size = 0;

	int    ray_index;
	Ray    ray;
	RayHit ray_hit;

	while (true) {
		bool inactive = stack_size == 0;

		if (inactive) {
			ray_index = atomic_agg_inc(rays_retired);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_trace.origin   .to_float3(ray_index);
			ray.direction = ray_buffer_trace.direction.to_float3(ray_index);
			ray.direction_inv = make_float3(
				1.0f / ray.direction.x, 
				1.0f / ray.direction.y, 
				1.0f / ray.direction.z
			);

			ray_hit.t           = INFINITY;
			ray_hit.triangle_id = -1;

			// Push root on stack
			stack_size                                = 1;
			shared_stack[threadIdx.x][threadIdx.y][0] = 1;
		}

		while (true) {
			assert(stack_size <= BVH_STACK_SIZE);

			unsigned packed = stack_pop(shared_stack, stack, stack_size);

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
						stack_push(shared_stack, stack, stack_size, pack_qbvh_node(child, id));
					}
				}
			}

			if (stack_size == 0) {
				ray_buffer_trace.triangle_id[ray_index] = ray_hit.triangle_id;
				ray_buffer_trace.u[ray_index] = ray_hit.u;
				ray_buffer_trace.v[ray_index] = ray_hit.v;

				break;
			}
		}
	}
}

__device__ inline void bvh_trace_shadow(int ray_count, int * rays_retired, int bounce) {
	__shared__ unsigned shared_stack[WARP_SIZE][SHADOW_TRACE_BLOCK_Y][SHARED_STACK_SIZE];

	unsigned stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int stack_size = 0;

	int ray_index;
	Ray ray;
	
	float max_distance;

	while (true) {
		bool inactive = stack_size == 0;

		if (inactive) {
			ray_index = atomic_agg_inc(rays_retired);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_shadow.ray_origin   .to_float3(ray_index);
			ray.direction = ray_buffer_shadow.ray_direction.to_float3(ray_index);
			ray.direction_inv = make_float3(
				1.0f / ray.direction.x, 
				1.0f / ray.direction.y, 
				1.0f / ray.direction.z
			);

			max_distance = ray_buffer_shadow.max_distance[ray_index];

			// Push root on stack
			stack_size                                = 1;
			shared_stack[threadIdx.x][threadIdx.y][0] = 1;
		}

		while (true) {
			// Pop Node of the stack
			unsigned packed = stack_pop(shared_stack, stack, stack_size);

			int node_index, node_id;
			unpack_qbvh_node(packed, node_index, node_id);

			int2 index_and_count = qbvh_nodes[node_index].index_and_count[node_id];

			int index = index_and_count.x;
			int count = index_and_count.y;

			ASSERT(index != -1 && count != -1, "Unpacked invalid Node!");

			// Check if the Node is a leaf
			if (count > 0) {
				bool hit = false;

				for (int j = index; j < index + count; j++) {
					if (triangle_trace_shadow(j, ray, max_distance)) {
						hit = true;

						break;
					}
				}

				if (hit) {
					stack_size = 0;

					break;
				}
			} else {
				int child = index;

				AABBHits aabb_hits = qbvh_node_intersect(qbvh_nodes[child], ray, max_distance);
				
				for (int i = 0; i < 4; i++) {
					// Extract index from the 2 least significant bits
					int id = aabb_hits.t_near_i[i] & 0b11;
					
					if (aabb_hits.hit[id]) {
						stack_push(shared_stack, stack, stack_size, pack_qbvh_node(child, id));
					}
				}
			}

			if (stack_size == 0) {
				// We didn't hit anything, apply illumination
				int    pixel_index  = ray_buffer_shadow.pixel_index[ray_index];
				float3 illumination = ray_buffer_shadow.illumination.to_float3(ray_index);

				if (bounce == 0) {
					frame_buffer_direct[pixel_index] += make_float4(illumination);
				} else {
					frame_buffer_indirect[pixel_index] += make_float4(illumination);
				}

				break;
			}
		}
	}
}
#elif BVH_TYPE == BVH_CWBVH
typedef unsigned char byte;

struct CWBVHNode {
	float4 node_0;
	float4 node_1;
	float4 node_2;
	float4 node_3;
	float4 node_4; // Node is stored as 5 float4's so we can load the entire 80 bytes in 5 global memory accesses
};

__device__ CWBVHNode * cwbvh_nodes;

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

		// Select near and far planes based on ray octant
		unsigned q_lo_x = float_as_uint(i == 0 ? node_2.x : node_2.y);
		unsigned q_hi_x = float_as_uint(i == 0 ? node_2.z : node_2.w);

		unsigned q_lo_y = float_as_uint(i == 0 ? node_3.x : node_3.y);
		unsigned q_hi_y = float_as_uint(i == 0 ? node_3.z : node_3.w);

		unsigned q_lo_z = float_as_uint(i == 0 ? node_4.x : node_4.y);
		unsigned q_hi_z = float_as_uint(i == 0 ? node_4.z : node_4.w);

		unsigned x_min = ray_negative_x ? q_hi_x : q_lo_x;
		unsigned x_max = ray_negative_x ? q_lo_x : q_hi_x;

		unsigned y_min = ray_negative_y ? q_hi_y : q_lo_y;
		unsigned y_max = ray_negative_y ? q_lo_y : q_hi_y;

		unsigned z_min = ray_negative_z ? q_hi_z : q_lo_z;
		unsigned z_max = ray_negative_z ? q_lo_z : q_hi_z;

		#pragma unroll
		for (int j = 0; j < 4; j++) {
			// Extract j-th bytes
			float3 tmin3 = make_float3(float(extract_byte(x_min, j)), float(extract_byte(y_min, j)), float(extract_byte(z_min, j)));
			float3 tmax3 = make_float3(float(extract_byte(x_max, j)), float(extract_byte(y_max, j)), float(extract_byte(z_max, j)));

			// Account for grid origin and scale
			tmin3 = tmin3 * adjusted_ray_direction_inv + adjusted_ray_origin;
			tmax3 = tmax3 * adjusted_ray_direction_inv + adjusted_ray_origin;

			float tmin = vmax_max(tmin3.x, tmin3.y, fmaxf(tmin3.z, EPSILON));
			float tmax = vmin_min(tmax3.x, tmax3.y, fminf(tmax3.z, max_distance));

			bool intersected = tmin < tmax;
			if (intersected) {
				unsigned child_bits = extract_byte(child_bits4, j);
				unsigned bit_index  = extract_byte(bit_index4,  j);

				hit_mask |= child_bits << bit_index;
			}
		}
	}

	return hit_mask;
}

// Constants used by Dynamic Fetch Heurisic (see section 4.4 of Ylitie et al. 2017)
#define N_d 4
#define N_w 16

__device__ inline void bvh_trace(int ray_count, int * rays_retired) {
	__shared__ uint2 shared_stack[WARP_SIZE][TRACE_BLOCK_Y][SHARED_STACK_SIZE];

	uint2 stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int   stack_size = 0;

	uint2 current_group = make_uint2(0, 0);

	int ray_index;
	Ray ray;

	bool ray_negative_x, ray_negative_y, ray_negative_z;

	unsigned oct_inv;
	unsigned oct_inv4;
	
	RayHit ray_hit;

	while (true) {
		bool inactive = stack_size == 0 && current_group.y <= 0x00ffffff;

		if (inactive) {
			ray_index = atomic_agg_inc(rays_retired);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_trace.origin   .to_float3(ray_index);
			ray.direction = ray_buffer_trace.direction.to_float3(ray_index);
			ray.direction_inv = make_float3(
				1.0f / ray.direction.x, 
				1.0f / ray.direction.y, 
				1.0f / ray.direction.z
			);

			ray_negative_x = ray.direction.x < 0.0f;
			ray_negative_y = ray.direction.y < 0.0f;
			ray_negative_z = ray.direction.z < 0.0f;

			unsigned oct = 
				(ray_negative_x ? 0b100 : 0) |
				(ray_negative_y ? 0b010 : 0) |
				(ray_negative_z ? 0b001 : 0);

			oct_inv  = 7 - oct;
			oct_inv4 = oct_inv * 0x01010101;

			current_group = make_uint2(0, 0x80000000);

			ray_hit.t           = INFINITY;
			ray_hit.triangle_id = -1;
		}

		int iterations_lost = 0;

		do {
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

					stack_push(shared_stack, stack, stack_size, current_group);
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

			// 20% of active threads in warp
			int utilization_threshold = __popc(active_thread_mask()) / 5;

			// While the triangle group is not empty
			while (triangle_group.y != 0) {
#if TRIANGLE_POSTPONING
				if (__popc(active_thread_mask()) < utilization_threshold) {
					stack_push(shared_stack, stack, stack_size, triangle_group);

					break;
				}
#endif

				int triangle_index = msb(triangle_group.y);

				triangle_group.y &= ~(1 << triangle_index);

				triangle_trace(triangle_group.x + triangle_index, ray, ray_hit);
			}

			if (current_group.y <= 0x00ffffff) {
				if (stack_size == 0) {
					ray_buffer_trace.triangle_id[ray_index] = ray_hit.triangle_id;
					ray_buffer_trace.u[ray_index] = ray_hit.u;
					ray_buffer_trace.v[ray_index] = ray_hit.v;

					break;
				}

				current_group = stack_pop(shared_stack, stack, stack_size);
			}

			iterations_lost += WARP_SIZE - __popc(active_thread_mask()) - N_d;
		} while (iterations_lost < N_w);
	}
}

__device__ inline void bvh_trace_shadow(int ray_count, int * rays_retired, int bounce) {
	__shared__ uint2 shared_stack[WARP_SIZE][SHADOW_TRACE_BLOCK_Y][SHARED_STACK_SIZE];

	uint2 stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int   stack_size = 0;

	uint2 current_group = make_uint2(0, 0);

	int ray_index;
	Ray ray;

	bool ray_negative_x, ray_negative_y, ray_negative_z;

	unsigned oct_inv;
	unsigned oct_inv4;

	float max_distance;

	while (true) {
		bool inactive = stack_size == 0 && current_group.y <= 0x00ffffff;

		if (inactive) {
			ray_index = atomic_agg_inc(rays_retired);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_shadow.ray_origin   .to_float3(ray_index);
			ray.direction = ray_buffer_shadow.ray_direction.to_float3(ray_index);
			ray.direction_inv = make_float3(
				1.0f / ray.direction.x, 
				1.0f / ray.direction.y, 
				1.0f / ray.direction.z
			);

			ray_negative_x = ray.direction.x < 0.0f;
			ray_negative_y = ray.direction.y < 0.0f;
			ray_negative_z = ray.direction.z < 0.0f;

			unsigned oct = 
				(ray_negative_x ? 0b100 : 0) |
				(ray_negative_y ? 0b010 : 0) |
				(ray_negative_z ? 0b001 : 0);

			oct_inv  = 7 - oct;
			oct_inv4 = oct_inv * 0x01010101;

			current_group = make_uint2(0, 0x80000000);

			max_distance = ray_buffer_shadow.max_distance[ray_index];
		}

		int iterations_lost = 0;

		do {
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

					// Push Stack
					if (stack_size < SHARED_STACK_SIZE) {
						shared_stack[threadIdx.x][threadIdx.y][stack_size] = current_group;
					} else {
						stack[stack_size - SHARED_STACK_SIZE] = current_group;
					}
					stack_size++;
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

			// 20% of active threads in warp
			int utilization_threshold = __popc(active_thread_mask()) / 5;

			bool hit = false;

			// While the triangle group is not empty
			while (triangle_group.y != 0) {
#if TRIANGLE_POSTPONING
				if (__popc(active_thread_mask()) < utilization_threshold) {
					stack_push(shared_stack, stack, stack_size, triangle_group);

					break;
				}
#endif

				int triangle_index = msb(triangle_group.y);

				triangle_group.y &= ~(1 << triangle_index);

				if (triangle_trace_shadow(triangle_group.x + triangle_index, ray, max_distance)) {
					hit = true;

					break;
				}
			}

			if (hit) {
				stack_size      = 0;
				current_group.y = 0;

				break;
			}

			if (current_group.y <= 0x00ffffff) {
				if (stack_size == 0) {
					// We didn't hit anything, apply illumination
					int    pixel_index  = ray_buffer_shadow.pixel_index[ray_index];
					float3 illumination = ray_buffer_shadow.illumination.to_float3(ray_index);

					if (bounce == 0) {
						frame_buffer_direct[pixel_index] += make_float4(illumination);
					} else {
						frame_buffer_indirect[pixel_index] += make_float4(illumination);
					}

					break;
				}

				// Pop Stack
				stack_size--;
				if (stack_size < SHARED_STACK_SIZE) {
					current_group = shared_stack[threadIdx.x][threadIdx.y][stack_size];
				} else {
					current_group = stack[stack_size - SHARED_STACK_SIZE];
				}
			}

			iterations_lost += WARP_SIZE - __popc(active_thread_mask()) - N_d;
		} while (iterations_lost < N_w);
	}
}
#endif
