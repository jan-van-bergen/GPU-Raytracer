#pragma once
#include "Triangle.h"

#include "CUDA_Source/Common.h"

struct BVHNode {
	AABB aabb;
	union {
		int left;
		int first;
	};
	int count; // Stores split axis in its 2 highest bits, count in its lowest 30 bits

	inline int get_count() const {
		return count & ~BVH_AXIS_MASK;
	}

	inline bool is_leaf() const {
		return get_count() > 0;
	}
};

struct BVH {
	int        triangle_count;
	Triangle * triangles;

	int   index_count;
	int * indices;

	int       node_count;
	BVHNode * nodes;
};
