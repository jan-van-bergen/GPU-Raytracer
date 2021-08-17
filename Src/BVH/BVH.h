#pragma once
#include "Pathtracer/Triangle.h"

#include "../CUDA_Source/Common.h"

typedef unsigned char byte;

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

struct QBVHNode {
	float aabb_min_x[4] = { 0.0f };
	float aabb_min_y[4] = { 0.0f };
	float aabb_min_z[4] = { 0.0f };
	float aabb_max_x[4] = { 0.0f };
	float aabb_max_y[4] = { 0.0f };
	float aabb_max_z[4] = { 0.0f };

	struct {
		int index;
		int count;
	} index_and_count[4];

	inline       int & get_index(int i)       { return index_and_count[i].index; }
	inline const int & get_index(int i) const { return index_and_count[i].index; }
	inline       int & get_count(int i)       { return index_and_count[i].count; }
	inline const int & get_count(int i) const { return index_and_count[i].count; }

	inline bool is_leaf(int i) { return get_count(i) > 0; }

	inline int get_child_count() const {
		int result = 0;

		for (int i = 0; i < 4; i++) {
			if (get_count(i) == -1) break;

			result++;
		}

		return result;
	}

};

static_assert(sizeof(QBVHNode) == 128);

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

template<typename NodeType>
struct BVHBase {
	int   index_count;
	int * indices = nullptr;

	int        node_count;
	NodeType * nodes = nullptr;
};

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
typedef BVHNode   BVHNodeType;
#elif BVH_TYPE == BVH_QBVH
typedef QBVHNode  BVHNodeType;
#elif BVH_TYPE == BVH_CWBVH
typedef CWBVHNode BVHNodeType;
#endif

typedef BVHBase<BVHNode>   BVH;
typedef BVHBase<QBVHNode>  QBVH;
typedef BVHBase<CWBVHNode> CWBVH;

typedef BVHBase<BVHNodeType> BVHType;
