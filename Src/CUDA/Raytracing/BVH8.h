#pragma once
#include "BVHCommon.h"

// Inverse of ray octant, encoded in 3 bits, duplicated for each byte
__device__ inline unsigned ray_get_octant_inv4(const float3 ray_direction) {
	return
		(ray_direction.x < 0.0f ? 0 : 0x04040404) |
		(ray_direction.y < 0.0f ? 0 : 0x02020202) |
		(ray_direction.z < 0.0f ? 0 : 0x01010101);
}

// Inverse of the percentage of active threads that triggers triangle postponing
// A value of 5 means that if less than 1/5 = 20% of the active threads want to
// intersect triangles we postpone the intersection test to decrease divergence within a Warp
#define BVH8_TRIANGLE_POSTPONING_THRESHOLD_DIVISOR 5

typedef unsigned char byte;

struct BVH8Node {
	float4 node_0;
	float4 node_1;
	float4 node_2;
	float4 node_3;
	float4 node_4; // Node is stored as 5 float4's so we can load the entire 80 bytes in 5 global memory accesses
};

__device__ __constant__ const BVH8Node * bvh8_nodes;

__device__ inline unsigned bvh8_node_intersect(
	const Ray & ray,
	unsigned oct_inv4,
	float max_distance,
	const float4 & node_0, const float4 & node_1, const float4 & node_2, const float4 & node_3, const float4 & node_4
) {
	float3 p = make_float3(node_0);

	unsigned e_imask = __float_as_uint(node_0.w);
	byte e_x = extract_byte(e_imask, 0);
	byte e_y = extract_byte(e_imask, 1);
	byte e_z = extract_byte(e_imask, 2);

	float3 adjusted_ray_direction_inv = make_float3(
		__uint_as_float(e_x << 23) / ray.direction.x,
		__uint_as_float(e_y << 23) / ray.direction.y,
		__uint_as_float(e_z << 23) / ray.direction.z
	);
	float3 adjusted_ray_origin = (p - ray.origin) / ray.direction;

	unsigned hit_mask = 0;

	#pragma unroll
	for (int i = 0; i < 2; i++) {
		unsigned meta4 = __float_as_uint(i == 0 ? node_1.z : node_1.w);

		unsigned is_inner4   = (meta4 & (meta4 << 1)) & 0x10101010;
		unsigned inner_mask4 = sign_extend_s8x4(is_inner4 << 3);
		unsigned bit_index4  = (meta4 ^ (oct_inv4 & inner_mask4)) & 0x1f1f1f1f;
		unsigned child_bits4 = (meta4 >> 5) & 0x07070707;

		// Select near and far planes based on ray octant
		unsigned q_lo_x = __float_as_uint(i == 0 ? node_2.x : node_2.y);
		unsigned q_hi_x = __float_as_uint(i == 0 ? node_2.z : node_2.w);

		unsigned q_lo_y = __float_as_uint(i == 0 ? node_3.x : node_3.y);
		unsigned q_hi_y = __float_as_uint(i == 0 ? node_3.z : node_3.w);

		unsigned q_lo_z = __float_as_uint(i == 0 ? node_4.x : node_4.y);
		unsigned q_hi_z = __float_as_uint(i == 0 ? node_4.z : node_4.w);

		unsigned x_min = ray.direction.x < 0.0f ? q_hi_x : q_lo_x;
		unsigned x_max = ray.direction.x < 0.0f ? q_lo_x : q_hi_x;

		unsigned y_min = ray.direction.y < 0.0f ? q_hi_y : q_lo_y;
		unsigned y_max = ray.direction.y < 0.0f ? q_lo_y : q_hi_y;

		unsigned z_min = ray.direction.z < 0.0f ? q_hi_z : q_lo_z;
		unsigned z_max = ray.direction.z < 0.0f ? q_lo_z : q_hi_z;

		#pragma unroll
		for (int j = 0; j < 4; j++) {
			// Extract j-th byte
			float3 tmin3 = make_float3(float(extract_byte(x_min, j)), float(extract_byte(y_min, j)), float(extract_byte(z_min, j)));
			float3 tmax3 = make_float3(float(extract_byte(x_max, j)), float(extract_byte(y_max, j)), float(extract_byte(z_max, j)));

			// Account for grid origin and scale
			tmin3 = tmin3 * adjusted_ray_direction_inv + adjusted_ray_origin;
			tmax3 = tmax3 * adjusted_ray_direction_inv + adjusted_ray_origin;

			float tmin = vmax_max(tmin3.x, tmin3.y, fmaxf(tmin3.z, 0.0f));
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

// Constants used by Dynamic Fetch Heuristic (see section 4.4 of Ylitie et al. 2017)
#define N_d 4
#define N_w 16

__device__ inline void bvh8_trace(int bounce, int ray_count, int * rays_retired) {
	extern __shared__ uint2 shared_stack_bvh8[];

	uint2 stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int   stack_size = 0;

	uint2 current_group = make_uint2(0, 0);

	int ray_index;
	Ray ray;
	Ray ray_untransformed;

	unsigned oct_inv4;

	RayHit ray_hit;

	int  tlas_stack_size;
	int  mesh_id;
	bool mesh_has_identity_transform;

	while (true) {
		bool inactive = stack_size == 0 && current_group.y == 0;

		if (inactive) {
			ray_index = atomicAdd(rays_retired, 1);
			if (ray_index >= ray_count) return;

			ray.origin    = get_ray_buffer_trace(bounce)->origin   .get(ray_index);
			ray.direction = get_ray_buffer_trace(bounce)->direction.get(ray_index);
			ray_untransformed = ray;

			oct_inv4 = ray_get_octant_inv4(ray.direction);

			current_group = make_uint2(0, 0x80000000);

			ray_hit.t           = INFINITY;
			ray_hit.triangle_id = INVALID;

			tlas_stack_size = INVALID;
		}

		int iterations_lost = 0;

		do {
			uint2 triangle_group;

			if (current_group.y & 0xff000000) {
				unsigned hits_imask = current_group.y;

				unsigned child_index_offset = msb(hits_imask);
				unsigned child_index_base   = current_group.x;

				// Remove n from current_group;
				current_group.y &= ~(1 << child_index_offset);

				// If the node group is not yet empty, push it on the stack
				if (current_group.y & 0xff000000) {
					// assert(stack_size < BVH_STACK_SIZE);

					stack_push(shared_stack_bvh8, stack, stack_size, current_group);
				}

				unsigned slot_index     = (child_index_offset - 24) ^ (oct_inv4 & 0xff);
				unsigned relative_index = __popc(hits_imask & ~(0xffffffff << slot_index));

				unsigned child_node_index = child_index_base + relative_index;

				float4 node_0 = __ldg(&bvh8_nodes[child_node_index].node_0);
				float4 node_1 = __ldg(&bvh8_nodes[child_node_index].node_1);
				float4 node_2 = __ldg(&bvh8_nodes[child_node_index].node_2);
				float4 node_3 = __ldg(&bvh8_nodes[child_node_index].node_3);
				float4 node_4 = __ldg(&bvh8_nodes[child_node_index].node_4);

				unsigned hitmask = bvh8_node_intersect(ray, oct_inv4, ray_hit.t, node_0, node_1, node_2, node_3, node_4);

				byte imask = extract_byte(__float_as_uint(node_0.w), 3);

				current_group .x = __float_as_uint(node_1.x); // Child    base offset
				triangle_group.x = __float_as_uint(node_1.y); // Triangle base offset

				current_group .y = (hitmask & 0xff000000) | unsigned(imask);
				triangle_group.y = (hitmask & 0x00ffffff);
			} else {
				triangle_group = current_group;
				current_group  = make_uint2(0);
			}

			int postpone_threshold = __popc(__activemask()) / BVH8_TRIANGLE_POSTPONING_THRESHOLD_DIVISOR;

			// While the triangle group is not empty
			while (triangle_group.y != 0) {
				if (tlas_stack_size == INVALID) {
					int mesh_offset = msb(triangle_group.y);
					triangle_group.y &= ~(1 << mesh_offset);

					mesh_id = triangle_group.x + mesh_offset;

					if (triangle_group.y != 0) {
						stack_push(shared_stack_bvh8, stack, stack_size, triangle_group);
					}
					if (current_group.y & 0xff000000) {
						stack_push(shared_stack_bvh8, stack, stack_size, current_group);
					}

					tlas_stack_size = stack_size;

					int root_index = bvh_get_mesh_root_index(mesh_id, mesh_has_identity_transform);
					
					// Optimization: if the Mesh has an identity transform, don't bother loading and transforming
					if (!mesh_has_identity_transform) {
						Matrix3x4 transform_inv = mesh_get_transform_inv(mesh_id);
						matrix3x4_transform_position (transform_inv, ray.origin);
						matrix3x4_transform_direction(transform_inv, ray.direction);

						oct_inv4 = ray_get_octant_inv4(ray.direction);
					}

					current_group = make_uint2(root_index, 0x80000000);

					break;
				} else {
					int thread_count = __popc(__activemask());
					if (thread_count < postpone_threshold) {
						// Not enough threads currently active that want to check triangle intersection, postpone by pushing on the stack
						stack_push(shared_stack_bvh8, stack, stack_size, triangle_group);

						break;
					}

					int triangle_index = msb(triangle_group.y);
					triangle_group.y &= ~(1 << triangle_index);

					triangle_intersect(mesh_id, triangle_group.x + triangle_index, ray, ray_hit);
				}
			}

			if ((current_group.y & 0xff000000) == 0) {
				if (stack_size == 0) {
					get_ray_buffer_trace(bounce)->hits.set(ray_index, ray_hit);

					current_group.y = 0;

					break;
				}

				if (stack_size == tlas_stack_size) {
					tlas_stack_size = INVALID;

					if (!mesh_has_identity_transform) {
						// Reset Ray to untransformed version
						ray = ray_untransformed;
						oct_inv4 = ray_get_octant_inv4(ray.direction);
					}
				}

				current_group = stack_pop(shared_stack_bvh8, stack, stack_size);
			}

			iterations_lost += WARP_SIZE - __popc(__activemask()) - N_d;
		} while (iterations_lost < N_w);
	}
}

__device__ inline void bvh8_trace_shadow(int bounce, int ray_count, int * rays_retired) {
	extern __shared__ uint2 shared_stack_bvh8[];

	uint2 stack[BVH_STACK_SIZE - SHARED_STACK_SIZE];
	int   stack_size = 0;

	uint2 current_group = make_uint2(0, 0);

	Ray ray;
	Ray ray_untransformed;

	unsigned oct_inv4;

	float max_distance;

	float3 illumination; 
	int    pixel_index;

	int  tlas_stack_size;
	int  mesh_id;
	bool mesh_has_identity_transform;

	while (true) {
		bool inactive = stack_size == 0 && current_group.y == 0;

		if (inactive) {
			int ray_index = atomicAdd(rays_retired, 1);
			if (ray_index >= ray_count) return;

			ray.origin    = ray_buffer_shadow.ray_origin   .get(ray_index);
			ray.direction = ray_buffer_shadow.ray_direction.get(ray_index);
			ray_untransformed = ray;

			oct_inv4 = ray_get_octant_inv4(ray.direction);

			current_group = make_uint2(0, 0x80000000);

			max_distance = ray_buffer_shadow.max_distance[ray_index];

			float4 illumination_and_pixel_index = ray_buffer_shadow.illumination_and_pixel_index[ray_index];
			illumination = make_float3(illumination_and_pixel_index);
			pixel_index  = __float_as_int(illumination_and_pixel_index.w);

			tlas_stack_size = INVALID;
		}

		int iterations_lost = 0;

		do {
			uint2 triangle_group;

			if (current_group.y & 0xff000000) {
				unsigned hits_imask = current_group.y;

				unsigned child_index_offset = msb(hits_imask);
				unsigned child_index_base   = current_group.x;

				// Remove n from current_group;
				current_group.y &= ~(1 << child_index_offset);

				// If the node group is not yet empty, push it on the stack
				if (current_group.y & 0xff000000) {
					// assert(stack_size < BVH_STACK_SIZE);

					stack_push(shared_stack_bvh8, stack, stack_size, current_group);
				}

				unsigned slot_index     = (child_index_offset - 24) ^ (oct_inv4 & 0xff);
				unsigned relative_index = __popc(hits_imask & ~(0xffffffff << slot_index));

				unsigned child_node_index = child_index_base + relative_index;

				float4 node_0 = bvh8_nodes[child_node_index].node_0;
				float4 node_1 = bvh8_nodes[child_node_index].node_1;
				float4 node_2 = bvh8_nodes[child_node_index].node_2;
				float4 node_3 = bvh8_nodes[child_node_index].node_3;
				float4 node_4 = bvh8_nodes[child_node_index].node_4;

				unsigned hitmask = bvh8_node_intersect(ray, oct_inv4, max_distance, node_0, node_1, node_2, node_3, node_4);

				byte imask = extract_byte(__float_as_uint(node_0.w), 3);

				current_group .x = __float_as_uint(node_1.x); // Child    base offset
				triangle_group.x = __float_as_uint(node_1.y); // Triangle base offset

				current_group .y = (hitmask & 0xff000000) | unsigned(imask);
				triangle_group.y = (hitmask & 0x00ffffff);
			} else {
				triangle_group = current_group;
				current_group  = make_uint2(0);
			}

			int postpone_threshold = __popc(__activemask()) / BVH8_TRIANGLE_POSTPONING_THRESHOLD_DIVISOR;

			bool hit = false;

			// While the triangle group is not empty
			while (triangle_group.y != 0) {
				if (tlas_stack_size == INVALID) {
					int mesh_offset = msb(triangle_group.y);
					triangle_group.y &= ~(1 << mesh_offset);

					mesh_id = triangle_group.x + mesh_offset;

					if (triangle_group.y != 0) {
						stack_push(shared_stack_bvh8, stack, stack_size, triangle_group);
					}
					if (current_group.y & 0xff000000) {
						stack_push(shared_stack_bvh8, stack, stack_size, current_group);
					}

					tlas_stack_size = stack_size;

					int root_index = bvh_get_mesh_root_index(mesh_id, mesh_has_identity_transform);
					
					// Optimization: if the Mesh has an identity transform, don't bother loading and transforming
					if (!mesh_has_identity_transform) {
						Matrix3x4 transform_inv = mesh_get_transform_inv(mesh_id);
						matrix3x4_transform_position (transform_inv, ray.origin);
						matrix3x4_transform_direction(transform_inv, ray.direction);

						oct_inv4 = ray_get_octant_inv4(ray.direction);
					}

					current_group = make_uint2(root_index, 0x80000000);

					break;
				} else {
					int thread_count = __popc(__activemask());
					if (thread_count < postpone_threshold) {
						// Not enough threads currently active that want to check triangle intersection, postpone by pushing on the stack
						stack_push(shared_stack_bvh8, stack, stack_size, triangle_group);

						break;
					}

					int triangle_index = msb(triangle_group.y);
					triangle_group.y &= ~(1 << triangle_index);

					if (triangle_intersect_shadow(triangle_group.x + triangle_index, ray, max_distance)) {
						hit = true;

						break;
					}
				}
			}

			if (hit) {
				stack_size      = 0;
				current_group.y = 0;

				break;
			}

			if ((current_group.y & 0xff000000) == 0) {
				if (stack_size == 0) {
					// We didn't hit anything, apply illumination
					if (bounce == 0) {
						frame_buffer_direct[pixel_index] += make_float4(illumination);
					} else {
						frame_buffer_indirect[pixel_index] += make_float4(illumination);
					}

					current_group.y = 0;

					break;
				}

				if (stack_size == tlas_stack_size) {
					tlas_stack_size = INVALID;

					if (!mesh_has_identity_transform) {
						// Reset Ray to untransformed version
						ray = ray_untransformed;
						oct_inv4 = ray_get_octant_inv4(ray.direction);
					}
				}

				current_group = stack_pop(shared_stack_bvh8, stack, stack_size);
			}

			iterations_lost += WARP_SIZE - __popc(__activemask()) - N_d;
		} while (iterations_lost < N_w);
	}
}