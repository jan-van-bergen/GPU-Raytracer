#pragma once
#include "BVH.h"
#include "BVHBuilders.h"

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

struct QBVH {
	int        triangle_count;
	Triangle * triangles;

	int * indices;

	int        node_count;
	QBVHNode * nodes;

	int leaf_count;

	inline void init(const BVH & bvh) {
		triangle_count = bvh.triangle_count;
		triangles      = bvh.triangles;

		indices = bvh.indices;

		node_count = bvh.node_count;
		nodes      = new QBVHNode[bvh.node_count];

		leaf_count = bvh.leaf_count;

		for (int i = 0; i < node_count; i++) {
			// We use index 1 as a starting point, such that it points to the first child of the root
			if (i == 1) {
				nodes[i].get_index(0) = 0;
				nodes[i].get_count(0) = 0;
			}

			if (!bvh.nodes[i].is_leaf()) {
				const BVHNode & child_left  = bvh.nodes[bvh.nodes[i].left];
				const BVHNode & child_right = bvh.nodes[bvh.nodes[i].left + 1];

				nodes[i].aabb_min_x[0] = child_left.aabb.min.x;
				nodes[i].aabb_min_y[0] = child_left.aabb.min.y;
				nodes[i].aabb_min_z[0] = child_left.aabb.min.z;
				nodes[i].aabb_max_x[0] = child_left.aabb.max.x;
				nodes[i].aabb_max_y[0] = child_left.aabb.max.y;
				nodes[i].aabb_max_z[0] = child_left.aabb.max.z;
				nodes[i].aabb_min_x[1] = child_right.aabb.min.x;
				nodes[i].aabb_min_y[1] = child_right.aabb.min.y;
				nodes[i].aabb_min_z[1] = child_right.aabb.min.z;
				nodes[i].aabb_max_x[1] = child_right.aabb.max.x;
				nodes[i].aabb_max_y[1] = child_right.aabb.max.y;
				nodes[i].aabb_max_z[1] = child_right.aabb.max.z;

				if (child_left.is_leaf()) {
					nodes[i].get_index(0) = child_left.first;
					nodes[i].get_count(0) = child_left.get_count();
				} else {
					nodes[i].get_index(0) = bvh.nodes[i].left;
					nodes[i].get_count(0) = 0;
				}

				if (child_right.is_leaf()) {
					nodes[i].get_index(1) = child_right.first;
					nodes[i].get_count(1) = child_right.get_count();
				} else {
					nodes[i].get_index(1) = bvh.nodes[i].left + 1;
					nodes[i].get_count(1) = 0;
				}
				
				// For now the tree is binary, 
				// so make the rest of the indices invalid
				for (int j = 2; j < 4; j++) {
					nodes[i].get_index(j) = -1;
					nodes[i].get_count(j) = -1;
				}
			}
		}

		// Collapse binary BVH into a QBVH
		BVHBuilders::qbvh_from_binary_bvh(nodes);
	}
};
