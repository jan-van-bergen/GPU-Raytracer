#include "BVHBuilders.h"

#include "QBVH.h"

static void collapse(QBVHNode nodes[], int node_index) {
	QBVHNode & node = nodes[node_index];

	while (true) {
		int child_count = node.get_child_count();

		// Look for adoptable child with the largest surface area
		float max_area  = -INFINITY;
		int   max_index = -1;

		for (int i = 0; i < child_count; i++) {
			// If child Node i is an internal node
			if (node.get_count(i) == 0) {
				int child_i_child_count = nodes[node.get_index(i)].get_child_count();

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

		const QBVHNode & max_child = nodes[node.get_index(max_index)];

		int max_child_child_count = max_child.get_child_count();

		// Replace max child Node with its first child
		node.aabb_min_x[max_index] = max_child.aabb_min_x[0];
		node.aabb_min_y[max_index] = max_child.aabb_min_y[0];
		node.aabb_min_z[max_index] = max_child.aabb_min_z[0];
		node.aabb_max_x[max_index] = max_child.aabb_max_x[0];
		node.aabb_max_y[max_index] = max_child.aabb_max_y[0];
		node.aabb_max_z[max_index] = max_child.aabb_max_z[0];
		node.get_index(max_index) = max_child.get_index(0);
		node.get_count(max_index) = max_child.get_count(0);

		// Add the rest of max child Node's children after the current Node's own children
		for (int i = 1; i < max_child_child_count; i++) {
			node.aabb_min_x[child_count + i - 1] = max_child.aabb_min_x[i];
			node.aabb_min_y[child_count + i - 1] = max_child.aabb_min_y[i];
			node.aabb_min_z[child_count + i - 1] = max_child.aabb_min_z[i];
			node.aabb_max_x[child_count + i - 1] = max_child.aabb_max_x[i];
			node.aabb_max_y[child_count + i - 1] = max_child.aabb_max_y[i];
			node.aabb_max_z[child_count + i - 1] = max_child.aabb_max_z[i];
			node.get_index (child_count + i - 1) = max_child.get_index (i);
			node.get_count (child_count + i - 1) = max_child.get_count (i);
		}
	};

	for (int i = 0; i < 4; i++) {
		if (node.get_count(i) == -1) break;

		// If child Node i is an internal node, recurse
		if (node.get_count(i) == 0) {
			collapse(nodes, node.get_index(i));
		}
	}
}

void BVHBuilders::qbvh_from_binary_bvh(QBVHNode nodes[]) {
	collapse(nodes, 0);
}
