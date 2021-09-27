#include "BVHCollapser.h"

#include "Util/BitArray.h"

struct CollapseCost {
	int primitive_count;
	float sah;
};

// Bottom up calculation of the cost of collapsing multiple leaf nodes into one
static CollapseCost calc_collapse_cost(const BVH & bvh, BitArray & collapse, int node_index = 0) {
	const BVHNode2 & node = bvh.nodes_2[node_index];

	if (node.is_leaf()) {
		return { int(node.count), float(node.count) * config.sah_cost_leaf };
	} else {
		CollapseCost cost_left  = calc_collapse_cost(bvh, collapse, node.left);
		CollapseCost cost_right = calc_collapse_cost(bvh, collapse, node.left + 1);

		int total_primtive_count = cost_left.primitive_count + cost_right.primitive_count;

		float sah_leaf = config.sah_cost_leaf * float(total_primtive_count);
		float sah_node = config.sah_cost_node + (
			bvh.nodes_2[node.left    ].aabb.surface_area() * cost_left .sah +
			bvh.nodes_2[node.left + 1].aabb.surface_area() * cost_right.sah
		) / node.aabb.surface_area();

		if (sah_leaf < sah_node) {
			assert(!collapse[node_index]);
			collapse[node_index] = true;

			return { total_primtive_count, sah_leaf };
		} else {
			return { total_primtive_count, sah_node };
		}
	}
}

// Helper method that collapses all subnodes in a given subtree into a single leaf Node
static int collapse_subtree(const BVH & bvh, BVH & new_bvh, int node_index) {
	const BVHNode2 & node = bvh.nodes_2[node_index];

	if (node.is_leaf()) {
		for (int i = 0; i < node.count; i++) {
			new_bvh.indices[new_bvh.index_count++] = bvh.indices[node.first + i];
		}

		return node.count;
	} else {
		int count_left  = collapse_subtree(bvh, new_bvh, node.left);
		int count_right = collapse_subtree(bvh, new_bvh, node.left + 1);

		return count_left + count_right;
	}
};

// Collapse leaf nodes based on precalculated cost
static void bvh_collapse(const BVH & bvh, BVH & new_bvh, int new_index, BitArray & collapse, int node_index = 0) {
	const BVHNode2 & node = bvh.nodes_2[node_index];

	BVHNode2 & new_node = new_bvh.nodes_2[new_index];
	new_node.aabb  = node.aabb;
	new_node.count = node.count;
	new_node.axis  = node.axis;

	if (node.is_leaf()) {
		new_node.first = new_bvh.index_count;

		for (int i = 0; i < node.count; i++) {
			new_bvh.indices[new_bvh.index_count++] = bvh.indices[node.first + i];
		}

		assert(new_node.is_leaf());
	} else {
		// Check if this internal Node needs to collapse its subtree into a leaf
		if (collapse[node_index]) {
			new_node.count = collapse_subtree(bvh, new_bvh, node_index);
			new_node.first = new_bvh.index_count - new_node.count;

			assert(new_node.is_leaf());
		} else {
			new_node.left = new_bvh.node_count;
			new_bvh.node_count += 2;

			assert(!new_node.is_leaf());

			bvh_collapse(bvh, new_bvh, new_node.left,     collapse, node.left);
			bvh_collapse(bvh, new_bvh, new_node.left + 1, collapse, node.left + 1);
		}
	}
}

void BVHCollapser::collapse(BVH & bvh) {
	// Calculate costs of collapse, and fill array with the decision to collapse, yes or no
	BitArray collapse = { };
	collapse.init(bvh.node_count);
	collapse.set_all(false);
	calc_collapse_cost(bvh, collapse);

	// Collapse BVH using a copy
	BVH collapsed_bvh = { };
	collapsed_bvh.nodes_2 = new BVHNode2[bvh.node_count];
	collapsed_bvh.indices = new int     [bvh.index_count];
	collapsed_bvh.node_count  = 2;
	collapsed_bvh.index_count = 0;

	bvh_collapse(bvh, collapsed_bvh, 0, collapse);

	assert(collapsed_bvh.node_count  <= bvh.node_count);
	assert(collapsed_bvh.index_count == bvh.index_count);

	delete [] bvh.nodes_2;
	delete [] bvh.indices;

	bvh = collapsed_bvh;

	collapse.free();
}
