#include "BVHCollapser.h"

#include "Core/BitArray.h"

struct CollapseCost {
	int   primitive_count;
	float sah;
};

// Bottom up calculation of the cost of collapsing multiple leaf nodes into one
static CollapseCost calc_collapse_cost(const BVH2 & bvh, BitArray & collapse, int node_index = 0) {
	const BVHNode2 & node = bvh.nodes[node_index];

	if (node.is_leaf()) {
		return { int(node.count), float(node.count) * config.sah_cost_leaf };
	} else {
		CollapseCost cost_left  = calc_collapse_cost(bvh, collapse, node.left);
		CollapseCost cost_right = calc_collapse_cost(bvh, collapse, node.left + 1);

		int total_primtive_count = cost_left.primitive_count + cost_right.primitive_count;

		float sah_leaf = config.sah_cost_leaf * float(total_primtive_count);
		float sah_node = config.sah_cost_node + (
			bvh.nodes[node.left    ].aabb.surface_area() * cost_left .sah +
			bvh.nodes[node.left + 1].aabb.surface_area() * cost_right.sah
		) / node.aabb.surface_area();

		if (sah_leaf < sah_node) {
			ASSERT(!collapse[node_index]);
			collapse[node_index] = true;

			return { total_primtive_count, sah_leaf };
		} else {
			return { total_primtive_count, sah_node };
		}
	}
}

// Helper method that collapses all subnodes in a given subtree into a single leaf Node
static int collapse_subtree(const BVH2 & bvh, BVH & new_bvh, int node_index) {
	const BVHNode2 & node = bvh.nodes[node_index];

	if (node.is_leaf()) {
		for (int i = 0; i < node.count; i++) {
			new_bvh.indices.push_back(bvh.indices[node.first + i]);
		}

		return node.count;
	} else {
		int count_left  = collapse_subtree(bvh, new_bvh, node.left);
		int count_right = collapse_subtree(bvh, new_bvh, node.left + 1);

		return count_left + count_right;
	}
};

// Collapse leaf nodes based on precalculated cost
static void bvh_collapse(const BVH2 & bvh, BVH2 & new_bvh, int new_index, BitArray & collapse, int node_index = 0) {
	const BVHNode2 & node = bvh.nodes[node_index];

	BVHNode2 & new_node = new_bvh.nodes[new_index];
	new_node.aabb  = node.aabb;
	new_node.count = node.count;
	new_node.axis  = node.axis;

	if (node.is_leaf()) {
		new_node.first = new_bvh.indices.size();

		for (int i = 0; i < node.count; i++) {
			new_bvh.indices.push_back(bvh.indices[node.first + i]);
		}

		ASSERT(new_node.is_leaf());
	} else {
		// Check if this internal Node needs to collapse its subtree into a leaf
		if (collapse[node_index]) {
			new_node.count = collapse_subtree(bvh, new_bvh, node_index);
			new_node.first = new_bvh.indices.size() - new_node.count;

			ASSERT(new_node.is_leaf());
		} else {
			new_node.left = new_bvh.nodes.size();
			new_bvh.nodes.emplace_back();
			new_bvh.nodes.emplace_back();

			ASSERT(!new_node.is_leaf());

			bvh_collapse(bvh, new_bvh, new_node.left,     collapse, node.left);
			bvh_collapse(bvh, new_bvh, new_node.left + 1, collapse, node.left + 1);
		}
	}
}

void BVHCollapser::collapse(BVH2 & bvh) {
	// Calculate costs of collapse, and fill array with the decision to collapse, yes or no
	BitArray collapse(bvh.nodes.size());
	collapse.set_all(false);
	calc_collapse_cost(bvh, collapse);

	// Collapse BVH using a copy
	BVH2 collapsed_bvh = { };
	collapsed_bvh.indices.reserve(bvh.indices.size());
	collapsed_bvh.nodes  .reserve(bvh.nodes  .size());

	collapsed_bvh.nodes.emplace_back(); // Root
	collapsed_bvh.nodes.emplace_back(); // Dummy

	bvh_collapse(bvh, collapsed_bvh, 0, collapse);

	ASSERT(collapsed_bvh.indices.size() == bvh.indices.size());
	ASSERT(collapsed_bvh.nodes  .size() <= bvh.nodes  .size());

	bvh = std::move(collapsed_bvh);
}
