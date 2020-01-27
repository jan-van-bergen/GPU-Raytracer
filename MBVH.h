#pragma once
#include "BVH.h"

struct MBVHNode {
	float aabb_min_x[MBVH_WIDTH];
	float aabb_min_y[MBVH_WIDTH];
	float aabb_min_z[MBVH_WIDTH];
	float aabb_max_x[MBVH_WIDTH];
	float aabb_max_y[MBVH_WIDTH];
	float aabb_max_z[MBVH_WIDTH];
	union {
		int index[MBVH_WIDTH];
		int child[MBVH_WIDTH];
	};
	int count[MBVH_WIDTH];

	inline int get_child_count() const {
		int result = 0;

		for (int i = 0; i < MBVH_WIDTH; i++) {
			if (count[i] == -1) break;

			result++;
		}

		return result;
	}

	inline void collapse(MBVHNode nodes[]) {
		while (true) {
			int child_count = get_child_count();

			// Look for adoptable child with the largest surface area
			float max_area  = -INFINITY;
			int   max_index = -1;

			for (int i = 0; i < child_count; i++) {
				// If child Node i is an internal node
				if (count[i] == 0) {
					int child_i_child_count = nodes[child[i]].get_child_count();

					// Check if the current Node can adopt the children of child Node i
					if (child_count + child_i_child_count - 1 <= MBVH_WIDTH) {
						float diff_x = aabb_max_x[i] - aabb_min_x[i];
						float diff_y = aabb_max_y[i] - aabb_min_y[i];
						float diff_z = aabb_max_z[i] - aabb_min_z[i];

						float half_area = diff_x * diff_y + diff_y * diff_z + diff_z * diff_x;

						if (half_area > max_area) {
							max_area  = half_area;
							max_index = i;
						}
					}
				}
			}

			// No merge possible anymore, stop trying
			if (max_index == -1) break;

			const MBVHNode & max_child = nodes[child[max_index]];
			int max_child_child_count = max_child.get_child_count();

			// Replace max child Node with its first child
			aabb_min_x[max_index] = max_child.aabb_min_x[0];
			aabb_min_y[max_index] = max_child.aabb_min_y[0];
			aabb_min_z[max_index] = max_child.aabb_min_z[0];
			aabb_max_x[max_index] = max_child.aabb_max_x[0];
			aabb_max_y[max_index] = max_child.aabb_max_y[0];
			aabb_max_z[max_index] = max_child.aabb_max_z[0];
			child[max_index] = max_child.child[0];
			count[max_index] = max_child.count[0];

			// Add the rest of max child Node's children after the current Node's own children
			for (int i = 1; i < max_child_child_count; i++) {
				aabb_min_x[child_count + i - 1] = max_child.aabb_min_x[i];
				aabb_min_y[child_count + i - 1] = max_child.aabb_min_y[i];
				aabb_min_z[child_count + i - 1] = max_child.aabb_min_z[i];
				aabb_max_x[child_count + i - 1] = max_child.aabb_max_x[i];
				aabb_max_y[child_count + i - 1] = max_child.aabb_max_y[i];
				aabb_max_z[child_count + i - 1] = max_child.aabb_max_z[i];
				child[child_count + i - 1] = max_child.child[i];
				count[child_count + i - 1] = max_child.count[i];
			}
		};

		for (int i = 0; i < MBVH_WIDTH; i++) {
			if (count[i] == -1) break;

			// If child Node i is an internal node, recurse
			if (count[i] == 0) {
				nodes[child[i]].collapse(nodes);
			}
		}
	}
};

template<typename PrimitiveType>
struct MBVH {
	int             primitive_count;
	PrimitiveType * primitives;

	int * indices;

	int        node_count;
	MBVHNode * nodes;

	int leaf_count;

	inline void init(const BVH<PrimitiveType> & bvh) {
		primitive_count = bvh.primitive_count;
		primitives      = bvh.primitives;

		indices = bvh.indices_x;

		node_count = bvh.node_count;
		nodes      = new MBVHNode[bvh.node_count];

		leaf_count = bvh.leaf_count;

		for (int i = 0; i < node_count; i++) {
			// We use index 1 as a starting point, such that it points to the first child of the root
			if (i == 1) {
				nodes[i].child[0] = 0;
				nodes[i].count[0] = 0;
			}

			if (bvh.nodes[i].is_leaf() == false) {
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
					nodes[i].index[0] = child_left.first;
					nodes[i].count[0] = child_left.get_count();
				} else {
					nodes[i].child[0] = bvh.nodes[i].left;
					nodes[i].count[0] = 0;
				}

				if (child_right.is_leaf()) {
					nodes[i].index[1] = child_right.first;
					nodes[i].count[1] = child_right.get_count();
				} else {
					nodes[i].child[1] = bvh.nodes[i].left + 1;
					nodes[i].count[1] = 0;
				}
				
				// For now the tree is binary, 
				// so make the rest of the indices invalid
				for (int j = 2; j < MBVH_WIDTH; j++) {
					nodes[i].child[j] = -1;
					nodes[i].count[j] = -1;
				}
			}
		}

		// Collapse binary BVH into an MBVH
		nodes[0].collapse(nodes);
	}
};
