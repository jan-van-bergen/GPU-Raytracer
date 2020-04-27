#pragma once
#include "BVHBuilders.h"

#include "BVH.h"

void BVHBuilders::build_bvh(BVHNode & node, const Triangle * triangles, int * indices[3], BVHNode nodes[], int & node_index, int first_index, int index_count, float * sah, int * temp) {
	node.aabb = BVHPartitions::calculate_bounds(triangles, indices[0], first_index, first_index + index_count);
		
	if (index_count < 3) {
		// Leaf Node, terminate recursion
		node.first = first_index;
		node.count = index_count;

		return;
	}
		
	node.left = node_index;
	node_index += 2;
		
	int split_dimension;
	float split_cost;
	int split_index = BVHPartitions::partition_sah(triangles, indices, first_index, index_count, sah, temp, split_dimension, split_cost);

	// Check SAH termination condition
	float parent_cost = node.aabb.surface_area() * float(index_count); 
	if (split_cost >= parent_cost) {
		node.first = first_index;
		node.count = index_count;

		return;
	}

	float split = triangles[indices[split_dimension][split_index]].get_position()[split_dimension];
	BVHPartitions::split_indices(triangles, indices, first_index, index_count, temp, split_dimension, split_index, split);

	node.count = (split_dimension + 1) << 30;

	int n_left  = split_index - first_index;
	int n_right = first_index + index_count - split_index;

	build_bvh(nodes[node.left    ], triangles, indices, nodes, node_index, first_index,          n_left,  sah, temp);
	build_bvh(nodes[node.left + 1], triangles, indices, nodes, node_index, first_index + n_left, n_right, sah, temp);
}
