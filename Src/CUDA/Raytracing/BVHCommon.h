#pragma once
#include "Mesh.h"
#include "Triangle.h"

#include "../Buffers.h"

#define SHARED_STACK_INDEX(offset) ((threadIdx.y * SHARED_STACK_SIZE + offset) * WARP_SIZE + threadIdx.x)

// Function that decides whether to push on the shared stack or thread local stack
template<typename T>
__device__ inline void stack_push(T shared_stack[], T stack[], int & stack_size, T item) {
	// assert(stack_size < BVH_STACK_SIZE);

	if (stack_size < SHARED_STACK_SIZE) {
		shared_stack[SHARED_STACK_INDEX(stack_size)] = item;
	} else {
		stack[stack_size - SHARED_STACK_SIZE] = item;
	}
	stack_size++;
}

// Function that decides whether to pop from the shared stack or thread local stack
template<typename T>
__device__ inline T stack_pop(const T shared_stack[], const T stack[], int & stack_size) {
	// assert(stack_size > 0);

	stack_size--;
	if (stack_size < SHARED_STACK_SIZE) {
		return shared_stack[SHARED_STACK_INDEX(stack_size)];
	} else {
		return stack[stack_size - SHARED_STACK_SIZE];
	}
}

__device__ inline int bvh_get_mesh_root_index(int mesh_id, bool & mesh_has_identity_transform) {
	unsigned root_index = __ldg(&mesh_bvh_root_indices[mesh_id]);

	mesh_has_identity_transform = root_index >> 31; // MSB stores whether the Mesh has an identity transform

	return root_index & 0x7fffffff;
}
