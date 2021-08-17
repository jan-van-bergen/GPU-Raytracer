#include "QBVHBuilder.h"

void QBVHBuilder::collapse(int node_index) {
	QBVHNode & node = qbvh->nodes[node_index];

	while (true) {
		int child_count = node.get_child_count();

		// Look for adoptable child with the largest surface area
		float max_area  = -INFINITY;
		int   max_index = -1;

		for (int i = 0; i < child_count; i++) {
			if (!node.is_leaf(i)) {
				int child_i_child_count = qbvh->nodes[node.get_index(i)].get_child_count();

				// Check if the current Node can adopt the children of child Node i
				if (child_count + child_i_child_count - 1 <= 4) {
					float diff_x = node.aabb_max_x[i] - node.aabb_min_x[i];
					float diff_y = node.aabb_max_y[i] - node.aabb_min_y[i];
					float diff_z = node.aabb_max_z[i] - node.aabb_min_z[i];

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

		const QBVHNode & max_child = qbvh->nodes[node.get_index(max_index)];

		// Replace max child Node with its first child
		node.aabb_min_x[max_index] = max_child.aabb_min_x[0];
		node.aabb_min_y[max_index] = max_child.aabb_min_y[0];
		node.aabb_min_z[max_index] = max_child.aabb_min_z[0];
		node.aabb_max_x[max_index] = max_child.aabb_max_x[0];
		node.aabb_max_y[max_index] = max_child.aabb_max_y[0];
		node.aabb_max_z[max_index] = max_child.aabb_max_z[0];
		node.get_index(max_index) = max_child.get_index(0);
		node.get_count(max_index) = max_child.get_count(0);

		int max_child_child_count = max_child.get_child_count();

		// Add the rest of max child Node's children after the current Node's own children
		for (int i = 1; i < max_child_child_count; i++) {
			node.aabb_min_x[child_count + i - 1] = max_child.aabb_min_x[i];
			node.aabb_min_y[child_count + i - 1] = max_child.aabb_min_y[i];
			node.aabb_min_z[child_count + i - 1] = max_child.aabb_min_z[i];
			node.aabb_max_x[child_count + i - 1] = max_child.aabb_max_x[i];
			node.aabb_max_y[child_count + i - 1] = max_child.aabb_max_y[i];
			node.aabb_max_z[child_count + i - 1] = max_child.aabb_max_z[i];
			node.get_index(child_count + i - 1) = max_child.get_index(i);
			node.get_count(child_count + i - 1) = max_child.get_count(i);
		}
	};

	for (int i = 0; i < 4; i++) {
		if (node.get_count(i) == -1) break;

		// If child Node i is an internal node, recurse
		if (node.get_count(i) == 0) {
			collapse(node.get_index(i));
		}
	}
}

void QBVHBuilder::build(const BVH & bvh) {
	for (int i = 0; i < qbvh->node_count; i++) {
		// We use index 1 as a starting point, such that it points to the first child of the root
		if (i == 1) {
			qbvh->nodes[i].get_index(0) = 0;
			qbvh->nodes[i].get_count(0) = 0;
		}

		if (!bvh.nodes[i].is_leaf()) {
			const BVHNode & child_left  = bvh.nodes[bvh.nodes[i].left];
			const BVHNode & child_right = bvh.nodes[bvh.nodes[i].left + 1];

			qbvh->nodes[i].aabb_min_x[0] = child_left.aabb.min.x;
			qbvh->nodes[i].aabb_min_y[0] = child_left.aabb.min.y;
			qbvh->nodes[i].aabb_min_z[0] = child_left.aabb.min.z;
			qbvh->nodes[i].aabb_max_x[0] = child_left.aabb.max.x;
			qbvh->nodes[i].aabb_max_y[0] = child_left.aabb.max.y;
			qbvh->nodes[i].aabb_max_z[0] = child_left.aabb.max.z;
			qbvh->nodes[i].aabb_min_x[1] = child_right.aabb.min.x;
			qbvh->nodes[i].aabb_min_y[1] = child_right.aabb.min.y;
			qbvh->nodes[i].aabb_min_z[1] = child_right.aabb.min.z;
			qbvh->nodes[i].aabb_max_x[1] = child_right.aabb.max.x;
			qbvh->nodes[i].aabb_max_y[1] = child_right.aabb.max.y;
			qbvh->nodes[i].aabb_max_z[1] = child_right.aabb.max.z;

			if (child_left.is_leaf()) {
				qbvh->nodes[i].get_index(0) = child_left.first;
				qbvh->nodes[i].get_count(0) = child_left.get_count();
			} else {
				qbvh->nodes[i].get_index(0) = bvh.nodes[i].left;
				qbvh->nodes[i].get_count(0) = 0;
			}

			if (child_right.is_leaf()) {
				qbvh->nodes[i].get_index(1) = child_right.first;
				qbvh->nodes[i].get_count(1) = child_right.get_count();
			} else {
				qbvh->nodes[i].get_index(1) = bvh.nodes[i].left + 1;
				qbvh->nodes[i].get_count(1) = 0;
			}

			// For now the tree is binary,
			// so make the rest of the indices invalid
			for (int j = 2; j < 4; j++) {
				qbvh->nodes[i].get_index(j) = -1;
				qbvh->nodes[i].get_count(j) = -1;
			}
		}
	}

	// Check for the special case where the root is a leaf
	if (bvh.nodes[0].is_leaf()) {
		qbvh->nodes[0].aabb_min_x[0] = bvh.nodes[0].aabb.min.x;
		qbvh->nodes[0].aabb_min_y[0] = bvh.nodes[0].aabb.min.y;
		qbvh->nodes[0].aabb_min_z[0] = bvh.nodes[0].aabb.min.z;
		qbvh->nodes[0].aabb_max_x[0] = bvh.nodes[0].aabb.max.x;
		qbvh->nodes[0].aabb_max_y[0] = bvh.nodes[0].aabb.max.y;
		qbvh->nodes[0].aabb_max_z[0] = bvh.nodes[0].aabb.max.z;

		qbvh->nodes[0].get_index(0) = bvh.nodes[0].first;
		qbvh->nodes[0].get_count(0) = bvh.nodes[0].get_count();

		for (int i = 1; i < 4; i++) {
			qbvh->nodes[0].get_index(i) = -1;
			qbvh->nodes[0].get_count(i) = -1;
		}
	} else {
		// Collapse tree top-down, starting from the root
		collapse(0);
	}
}
