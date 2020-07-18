#pragma once
#include "BVH.h"

#include "ScopeTimer.h"

typedef unsigned char byte;

struct CWBVHNode {
	Vector3 p;
	byte e[3];
	byte imask;

	unsigned base_index_child;
	unsigned base_index_triangle;

	byte meta[8] = { };

	byte quantized_min_x[8] = { }, quantized_max_x[8] = { };
	byte quantized_min_y[8] = { }, quantized_max_y[8] = { };
	byte quantized_min_z[8] = { }, quantized_max_z[8] = { };

	inline bool is_leaf(int child_index) {
		return (meta[child_index] & 0b00011111) < 24;
	}
};

static_assert(sizeof(CWBVHNode) == 80);

struct CWBVH {
	int   index_count;
	int * indices;

	int         node_count;
	CWBVHNode * nodes;
};
