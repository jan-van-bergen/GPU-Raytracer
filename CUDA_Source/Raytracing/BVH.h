#pragma once
#include "BVHCommon.h"

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

__device__ __constant__ BVHNode * bvh_nodes;

__device__ void bvh_trace(int ray_count, int * rays_retired) {
	extern __shared__ int shared_stack_bvh[];

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
			ray_index = atomic_agg_inc(rays_retired);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_trace.origin   .get(ray_index);
			ray.direction = ray_buffer_trace.direction.get(ray_index);
			ray.calc_direction_inv();

			ray_hit.t           = INFINITY;
			ray_hit.triangle_id = INVALID;

			tlas_stack_size = INVALID;

			// Push root on stack
			stack_size                              = 1;
			shared_stack_bvh[SHARED_STACK_INDEX(0)] = 0;
		}

		while (true) {
			if (stack_size == tlas_stack_size) {
				tlas_stack_size = INVALID;

				if (!mesh_has_identity_transform) {
					// Reset Ray to untransformed version
					ray.origin    = ray_buffer_trace.origin   .get(ray_index);
					ray.direction = ray_buffer_trace.direction.get(ray_index);
					ray.calc_direction_inv();
				}
			}

			// Pop Node of the stack
			int node_index = stack_pop(shared_stack_bvh, stack, stack_size);

			const BVHNode & node = bvh_nodes[node_index];

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

							ray.calc_direction_inv();
						}

						stack_push(shared_stack_bvh, stack, stack_size, root_index);
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

					stack_push(shared_stack_bvh, stack, stack_size, second);
					stack_push(shared_stack_bvh, stack, stack_size, first);
				}
			}

			if (stack_size == 0) {
				ray_buffer_trace.hits.set(ray_index, ray_hit);

				break;
			}
		}
	}
}

__device__ void bvh_trace_shadow(int ray_count, int * rays_retired, int bounce) {
	extern __shared__ int shared_stack_bvh[];

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
			ray_index = atomic_agg_inc(rays_retired);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_shadow.ray_origin   .get(ray_index);
			ray.direction = ray_buffer_shadow.ray_direction.get(ray_index);
			ray.calc_direction_inv();

			max_distance = ray_buffer_shadow.max_distance[ray_index];

			tlas_stack_size = INVALID;

			// Push root on stack
			stack_size                              = 1;
			shared_stack_bvh[SHARED_STACK_INDEX(0)] = 0;
		}

		while (true) {
			if (stack_size == tlas_stack_size) {
				tlas_stack_size = INVALID;

				if (!mesh_has_identity_transform) {
					// Reset Ray to untransformed version
					ray.origin    = ray_buffer_shadow.ray_origin   .get(ray_index);
					ray.direction = ray_buffer_shadow.ray_direction.get(ray_index);
					ray.calc_direction_inv();
				}
			}

			// Pop Node of the stack
			int node_index = stack_pop(shared_stack_bvh, stack, stack_size);

			const BVHNode & node = bvh_nodes[node_index];

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

							ray.calc_direction_inv();
						}

						stack_push(shared_stack_bvh, stack, stack_size, root_index);
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

					stack_push(shared_stack_bvh, stack, stack_size, second);
					stack_push(shared_stack_bvh, stack, stack_size, first);
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
