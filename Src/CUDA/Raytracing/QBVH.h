#pragma once
#include "BVHCommon.h"

struct QBVHNode {
	float4 aabb_min_x;
	float4 aabb_min_y;
	float4 aabb_min_z;
	float4 aabb_max_x;
	float4 aabb_max_y;
	float4 aabb_max_z;
	int2 index_and_count[4];
};

__device__ __constant__ QBVHNode * qbvh_nodes;

struct AABBHits {
	float t_near[4];
	bool  hit[4];
};

// Check the Ray agains the four AABB's of the children of the given QBVH Node
__device__ inline AABBHits qbvh_node_intersect(const QBVHNode & node, const Ray & ray, float max_distance) {
	AABBHits result;

	float4 tx0 = (__ldg(&node.aabb_min_x) - ray.origin.x) / ray.direction.x;
	float4 tx1 = (__ldg(&node.aabb_max_x) - ray.origin.x) / ray.direction.x;
	float4 ty0 = (__ldg(&node.aabb_min_y) - ray.origin.y) / ray.direction.y;
	float4 ty1 = (__ldg(&node.aabb_max_y) - ray.origin.y) / ray.direction.y;
	float4 tz0 = (__ldg(&node.aabb_min_z) - ray.origin.z) / ray.direction.z;
	float4 tz1 = (__ldg(&node.aabb_max_z) - ray.origin.z) / ray.direction.z;

	result.t_near[0] = vmin_max(tx0.x, tx1.x, vmin_max(ty0.x, ty1.x, vmin_max(tz0.x, tz1.x, 0.0f)));
	result.t_near[1] = vmin_max(tx0.y, tx1.y, vmin_max(ty0.y, ty1.y, vmin_max(tz0.y, tz1.y, 0.0f)));
	result.t_near[2] = vmin_max(tx0.z, tx1.z, vmin_max(ty0.z, ty1.z, vmin_max(tz0.z, tz1.z, 0.0f)));
	result.t_near[3] = vmin_max(tx0.w, tx1.w, vmin_max(ty0.w, ty1.w, vmin_max(tz0.w, tz1.w, 0.0f)));
	
	float4 t_far = make_float4(
		vmax_min(tx0.x, tx1.x, vmax_min(ty0.x, ty1.x, vmax_min(tz0.x, tz1.x, max_distance))),
		vmax_min(tx0.y, tx1.y, vmax_min(ty0.y, ty1.y, vmax_min(tz0.y, tz1.y, max_distance))),
		vmax_min(tx0.z, tx1.z, vmax_min(ty0.z, ty1.z, vmax_min(tz0.z, tz1.z, max_distance))),
		vmax_min(tx0.w, tx1.w, vmax_min(ty0.w, ty1.w, vmax_min(tz0.w, tz1.w, max_distance)))
	);

	result.hit[0] = result.t_near[0] < t_far.x;
	result.hit[1] = result.t_near[1] < t_far.y;
	result.hit[2] = result.t_near[2] < t_far.z;
	result.hit[3] = result.t_near[3] < t_far.w;

	// Use the two least significant bits of the float to store the index
	result.t_near[0] = __uint_as_float((__float_as_uint(result.t_near[0]) & 0xfffffffc) | 0);
	result.t_near[1] = __uint_as_float((__float_as_uint(result.t_near[1]) & 0xfffffffc) | 1);
	result.t_near[2] = __uint_as_float((__float_as_uint(result.t_near[2]) & 0xfffffffc) | 2);
	result.t_near[3] = __uint_as_float((__float_as_uint(result.t_near[3]) & 0xfffffffc) | 3);

	// Bubble sort to order the hit distances
	#pragma unroll
	for (int i = 1; i < 4; i++) {
		#pragma unroll
		for (int j = i - 1; j >= 0; j--) {
			if (result.t_near[j] < result.t_near[j + 1]) {
				swap(result.t_near[j], result.t_near[j + 1]);
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

__device__ inline void qbvh_trace(int bounce, int ray_count, int * rays_retired) {
	extern __shared__ unsigned shared_stack_qbvh[];

	unsigned stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int stack_size = 0;

	int    ray_index;
	Ray    ray;
	RayHit ray_hit;

	int  tlas_stack_size;
	int  mesh_id;
	bool mesh_has_identity_transform;

	while (true) {
		bool inactive = stack_size == 0;

		if (inactive) {
			ray_index = atomicAdd(rays_retired, 1);
			if (ray_index >= ray_count) return;

			ray.origin    = get_ray_buffer_trace(bounce)->origin   .get(ray_index);
			ray.direction = get_ray_buffer_trace(bounce)->direction.get(ray_index);

			ray_hit.t           = INFINITY;
			ray_hit.triangle_id = INVALID;

			tlas_stack_size = INVALID;

			// Push root on stack
			stack_size                               = 1;
			shared_stack_qbvh[SHARED_STACK_INDEX(0)] = 1;
		}

		while (true) {
			if (stack_size == tlas_stack_size) {
				tlas_stack_size = INVALID;

				if (!mesh_has_identity_transform) {
					// Reset Ray to untransformed version
					ray.origin    = get_ray_buffer_trace(bounce)->origin   .get(ray_index);
					ray.direction = get_ray_buffer_trace(bounce)->direction.get(ray_index);
				}
			}

			//assert(stack_size <= BVH_STACK_SIZE);

			unsigned packed = stack_pop(shared_stack_qbvh, stack, stack_size);

			int node_index, node_id;
			unpack_qbvh_node(packed, node_index, node_id);

			int2 index_and_count = __ldg(&qbvh_nodes[node_index].index_and_count[node_id]);

			int index = index_and_count.x;
			int count = index_and_count.y;

			ASSERT(index != INVALID && count != INVALID, "Unpacked invalid Node!");

			// Check if the Node is a leaf
			if (count > 0) {
				if (tlas_stack_size == INVALID) {
					tlas_stack_size = stack_size;

					mesh_id = index;

					unsigned root_index = bvh_get_mesh_root_index(mesh_id, mesh_has_identity_transform) + 1;

					if (!mesh_has_identity_transform) {
						Matrix3x4 transform_inv = mesh_get_transform_inv(mesh_id);
						matrix3x4_transform_position (transform_inv, ray.origin);
						matrix3x4_transform_direction(transform_inv, ray.direction);
					}

					stack_push(shared_stack_qbvh, stack, stack_size, root_index);
				} else {
					for (int j = index; j < index + count; j++) {
						triangle_intersect(mesh_id, j, ray, ray_hit);
					}
				}
			} else {
				int child = index;

				AABBHits aabb_hits = qbvh_node_intersect(qbvh_nodes[child], ray, ray_hit.t);

				for (int i = 0; i < 4; i++) {
					// Extract index from the 2 least significant bits
					int id = __float_as_uint(aabb_hits.t_near[i]) & 3;

					if (aabb_hits.hit[id]) {
						stack_push(shared_stack_qbvh, stack, stack_size, pack_qbvh_node(child, id));
					}
				}
			}

			if (stack_size == 0) {
				get_ray_buffer_trace(bounce)->hits.set(ray_index, ray_hit);

				break;
			}
		}
	}
}

__device__ inline void qbvh_trace_shadow(int bounce, int ray_count, int * rays_retired) {
	extern __shared__ unsigned shared_stack_qbvh[];

	unsigned stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int stack_size = 0;

	int ray_index;
	Ray ray;

	float max_distance;

	int  tlas_stack_size;
	int  mesh_id;
	bool mesh_has_identity_transform;

	while (true) {
		bool inactive = stack_size == 0;

		if (inactive) {
			ray_index = atomicAdd(rays_retired, 1);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_shadow.ray_origin   .get(ray_index);
			ray.direction = ray_buffer_shadow.ray_direction.get(ray_index);

			max_distance = ray_buffer_shadow.max_distance[ray_index];

			tlas_stack_size = INVALID;

			// Push root on stack
			stack_size                               = 1;
			shared_stack_qbvh[SHARED_STACK_INDEX(0)] = 1;
		}

		while (true) {
			if (stack_size == tlas_stack_size) {
				tlas_stack_size = INVALID;

				if (!mesh_has_identity_transform) {
					// Reset Ray to untransformed version
					ray.origin    = ray_buffer_shadow.ray_origin   .get(ray_index);
					ray.direction = ray_buffer_shadow.ray_direction.get(ray_index);
				}
			}

			// Pop Node of the stack
			unsigned packed = stack_pop(shared_stack_qbvh, stack, stack_size);

			int node_index, node_id;
			unpack_qbvh_node(packed, node_index, node_id);

			int2 index_and_count = qbvh_nodes[node_index].index_and_count[node_id];

			int index = index_and_count.x;
			int count = index_and_count.y;

			ASSERT(index != INVALID && count != INVALID, "Unpacked invalid Node!");

			// Check if the Node is a leaf
			if (count > 0) {
				if (tlas_stack_size == INVALID) {
					tlas_stack_size = stack_size;

					mesh_id = index;

					unsigned root_index = bvh_get_mesh_root_index(mesh_id, mesh_has_identity_transform) + 1;

					if (!mesh_has_identity_transform) {
						Matrix3x4 transform_inv = mesh_get_transform_inv(mesh_id);
						matrix3x4_transform_position (transform_inv, ray.origin);
						matrix3x4_transform_direction(transform_inv, ray.direction);
					}

					stack_push(shared_stack_qbvh, stack, stack_size, root_index);
				} else {
					bool hit = false;

					for (int j = index; j < index + count; j++) {
						if (triangle_intersect_shadow(j, ray, max_distance)) {
							hit = true;

							break;
						}
					}

					if (hit) {
						stack_size = 0;

						break;
					}
				}
			} else {
				int child = index;

				AABBHits aabb_hits = qbvh_node_intersect(qbvh_nodes[child], ray, max_distance);

				for (int i = 0; i < 4; i++) {
					// Extract index from the 2 least significant bits
					int id = __float_as_uint(aabb_hits.t_near[i]) & 3;

					if (aabb_hits.hit[id]) {
						stack_push(shared_stack_qbvh, stack, stack_size, pack_qbvh_node(child, id));
					}
				}
			}

			if (stack_size == 0) {
				// We didn't hit anything, apply illumination
				float4 illumination_and_pixel_index = ray_buffer_shadow.illumination_and_pixel_index[ray_index];
				float3 illumination = make_float3(illumination_and_pixel_index);
				int    pixel_index  = __float_as_int(illumination_and_pixel_index.w);

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