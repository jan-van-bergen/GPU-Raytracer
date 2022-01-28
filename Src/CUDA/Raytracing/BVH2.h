#pragma once
#include "BVH.h"

struct AABB {
	float3 min;
	float3 max;

	__device__ inline bool intersects(const Ray & ray, float max_distance) const {
		float3 t0 = (min - ray.origin) / ray.direction;
		float3 t1 = (max - ray.origin) / ray.direction;

		float t_near = vmin_max(t0.x, t1.x, vmin_max(t0.y, t1.y, vmin_max(t0.z, t1.z, 0.0f)));
		float t_far  = vmax_min(t0.x, t1.x, vmax_min(t0.y, t1.y, vmax_min(t0.z, t1.z, max_distance)));

		return t_near < t_far;
	}
};

struct BVH2Node {
	AABB aabb;
	union {
		int left;
		int first;
	};
	unsigned count : 30;
	unsigned axis  : 2;

	__device__ inline bool is_leaf() const {
		return count > 0;
	}

	__device__ inline bool should_visit_left_first(const Ray & ray) const {
		switch (axis) {
			case 0: return ray.direction.x > 0.0f;
			case 1: return ray.direction.y > 0.0f;
			case 2: return ray.direction.z > 0.0f;
		}

		assert(false && "Invalid BVH axis!");
		return false;
	}
};

__device__ __constant__ BVH2Node * bvh2_nodes;

__device__ void bvh2_trace(TraversalData * traversal_data, int ray_count, int * rays_retired) {
	extern __shared__ int shared_stack_bvh2[];

	int stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
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

			ray.origin    = traversal_data->ray_origin   .get(ray_index);
			ray.direction = traversal_data->ray_direction.get(ray_index);

			ray_hit.t           = INFINITY;
			ray_hit.triangle_id = INVALID;

			tlas_stack_size = INVALID;

			// Push root on stack
			stack_size                              = 1;
			shared_stack_bvh2[SHARED_STACK_INDEX(0)] = 0;
		}

		while (true) {
			if (stack_size == tlas_stack_size) {
				tlas_stack_size = INVALID;

				if (!mesh_has_identity_transform) {
					// Reset Ray to untransformed version
					ray.origin    = traversal_data->ray_origin   .get(ray_index);
					ray.direction = traversal_data->ray_direction.get(ray_index);
				}
			}

			// Pop Node of the stack
			int node_index = stack_pop(shared_stack_bvh2, stack, stack_size);

			const BVH2Node & node = bvh2_nodes[node_index];

			if (node.aabb.intersects(ray, ray_hit.t)) {
				if (node.is_leaf()) {
					if (tlas_stack_size == INVALID) {
						tlas_stack_size = stack_size;

						mesh_id = node.first;

						int root_index = bvh_get_mesh_root_index(mesh_id, mesh_has_identity_transform);

						if (!mesh_has_identity_transform) {
							Matrix3x4 transform_inv = mesh_get_transform_inv(mesh_id);
							matrix3x4_transform_position (transform_inv, ray.origin);
							matrix3x4_transform_direction(transform_inv, ray.direction);
						}

						stack_push(shared_stack_bvh2, stack, stack_size, root_index);
					} else {
						for (int i = node.first; i < node.first + node.count; i++) {
							triangle_intersect(mesh_id, i, ray, ray_hit);
						}
					}
				} else {
					int first, second;

					if (node.should_visit_left_first(ray)) {
						second = node.left + 1;
						first  = node.left;
					} else {
						second = node.left;
						first  = node.left + 1;
					}

					stack_push(shared_stack_bvh2, stack, stack_size, second);
					stack_push(shared_stack_bvh2, stack, stack_size, first);
				}
			}

			if (stack_size == 0) {
				traversal_data->hits.set(ray_index, ray_hit);
				break;
			}
		}
	}
}

template<typename OnMissCallback>
__device__ void bvh2_trace_shadow(ShadowTraversalData * traveral_data, int ray_count, int * rays_retired, OnMissCallback callback) {
	extern __shared__ int shared_stack_bvh2[];

	int stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
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

			ray.origin    = traveral_data->ray_origin   .get(ray_index);
			ray.direction = traveral_data->ray_direction.get(ray_index);

			max_distance = traveral_data->max_distance[ray_index];

			tlas_stack_size = INVALID;

			// Push root on stack
			stack_size                              = 1;
			shared_stack_bvh2[SHARED_STACK_INDEX(0)] = 0;
		}

		while (true) {
			if (stack_size == tlas_stack_size) {
				tlas_stack_size = INVALID;

				if (!mesh_has_identity_transform) {
					// Reset Ray to untransformed version
					ray.origin    = traveral_data->ray_origin   .get(ray_index);
					ray.direction = traveral_data->ray_direction.get(ray_index);
				}
			}

			// Pop Node of the stack
			int node_index = stack_pop(shared_stack_bvh2, stack, stack_size);

			const BVH2Node & node = bvh2_nodes[node_index];

			if (node.aabb.intersects(ray, max_distance)) {
				if (node.is_leaf()) {
					if (tlas_stack_size == INVALID) {
						tlas_stack_size = stack_size;

						mesh_id = node.first;

						int root_index = bvh_get_mesh_root_index(mesh_id, mesh_has_identity_transform);

						if (!mesh_has_identity_transform) {
							Matrix3x4 transform_inv = mesh_get_transform_inv(mesh_id);
							matrix3x4_transform_position (transform_inv, ray.origin);
							matrix3x4_transform_direction(transform_inv, ray.direction);
						}

						stack_push(shared_stack_bvh2, stack, stack_size, root_index);
					} else {
						bool hit = false;
						for (int i = node.first; i < node.first + node.count; i++) {
							if (triangle_intersect_shadow(i, ray, max_distance)) {
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
					int first, second;

					if (node.should_visit_left_first(ray)) {
						second = node.left + 1;
						first  = node.left;
					} else {
						second = node.left;
						first  = node.left + 1;
					}

					stack_push(shared_stack_bvh2, stack, stack_size, second);
					stack_push(shared_stack_bvh2, stack, stack_size, first);
				}
			}

			if (stack_size == 0) {
				// We didn't hit anything, call callback
				callback(ray_index);
				break;
			}
		}
	}
}
