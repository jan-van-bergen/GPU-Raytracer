#pragma once
#include <algorithm>

#include "BVHBuilders.h"

#define BVH_TRAVERSE_BRUTE_FORCE  0 // Doesn't use the tree structure of the BVH, checks every Primitive for every Ray
#define BVH_TRAVERSE_TREE_NAIVE   1 // Traverses the BVH in a naive way, always checking the left Node before the right Node
#define BVH_TRAVERSE_TREE_ORDERED 2 // Traverses the BVH based on the split axis and the direction of the Ray

#define BVH_TRAVERSAL_STRATEGY BVH_TRAVERSE_TREE_ORDERED

#define BVH_AXIS_X_BITS 0x40000000 // 01 00 zeroes...
#define BVH_AXIS_Y_BITS 0x80000000 // 10 00 zeroes...
#define BVH_AXIS_Z_BITS 0xc0000000 // 11 00 zeroes...
#define BVH_AXIS_MASK   0xc0000000 // 11 00 zeroes...

struct BVHNode {
	AABB aabb;
	int left_or_first;
	int count; // Stores split axis in its 2 highest bits, count in its lowest 30 bits

	inline bool is_leaf() const {
		return (count & (~BVH_AXIS_MASK)) > 0;
	}
};

template<typename PrimitiveType>
struct BVH {
	PrimitiveType * primitives;
	int             primitive_count;

	int * indices_x;
	int * indices_y;
	int * indices_z;

	int       node_count;
	BVHNode * nodes;

	inline void init(int count) {
		assert(count > 0);

		primitive_count = count; 
		primitives = new PrimitiveType[primitive_count];

		int overallocation = 2; // SBVH requires more space

		// Construct index array
		int * all_indices  = new int[3 * overallocation * primitive_count];
		indices_x = all_indices;
		indices_y = all_indices + primitive_count * overallocation;
		indices_z = all_indices + primitive_count * overallocation * 2;

		for (int i = 0; i < primitive_count; i++) {
			indices_x[i] = i;
			indices_y[i] = i;
			indices_z[i] = i;
		}

		// Construct Node pool
		nodes = reinterpret_cast<BVHNode *>(ALLIGNED_MALLOC(2 * primitive_count * sizeof(BVHNode), 64));
		assert((unsigned long long)nodes % 64 == 0);
	}

	inline void build_bvh() {
		float * sah = new float[primitive_count];
		
		std::sort(indices_x, indices_x + primitive_count, [&](int a, int b) { return primitives[a].get_position().x < primitives[b].get_position().x; });
		std::sort(indices_y, indices_y + primitive_count, [&](int a, int b) { return primitives[a].get_position().y < primitives[b].get_position().y; });
		std::sort(indices_z, indices_z + primitive_count, [&](int a, int b) { return primitives[a].get_position().z < primitives[b].get_position().z; });
		
		int * indices[3] = { indices_x, indices_y, indices_z };

		int * temp = new int[primitive_count];

		int node_index = 2;
		BVHBuilders::build_bvh(nodes[0], primitives, indices, nodes, node_index, 0, primitive_count, sah, temp);

		assert(node_index <= 2 * primitive_count);

		node_count = node_index;

		delete [] temp;
		delete [] sah;
		
		PrimitiveType * new_primitives = new PrimitiveType[primitive_count];

		for (int i = 0; i < primitive_count; i++) {
			new_primitives[i] = primitives[indices_x[i]];
		}

		delete [] primitives;
		primitives = new_primitives;
	}
	
	//inline void update() const {
	//	for (int i = 0; i < primitive_count; i++) {
	//		primitives[i].update();
	//	}
	//}
};
