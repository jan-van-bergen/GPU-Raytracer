#include "BVHBuilders.h"

#include <algorithm>

#include "BVH.h"
#include "BVHPartitions.h"

#include "ScopeTimer.h"

template<typename Primitive>
static void build_bvh(BVHNode & node, const Primitive * primitives, int * indices[3], BVHNode nodes[], int & node_index, int first_index, int index_count, float * sah, int * temp, int max_primitives_in_leaf) {
	node.aabb = BVHPartitions::calculate_bounds(primitives, indices[0], first_index, first_index + index_count);
		
	if (index_count == 1) {
		// Leaf Node, terminate recursion
		node.first = first_index;
		node.count = index_count;

		return;
	}
		
	node.left = node_index;
	node_index += 2;
		
	int split_dimension;
	float split_cost;
	int split_index = BVHPartitions::partition_sah(primitives, indices, first_index, index_count, sah, split_dimension, split_cost);

	if (index_count <= max_primitives_in_leaf){
		// Check SAH termination condition
		float parent_cost = node.aabb.surface_area() * float(index_count); 
		if (split_cost >= parent_cost) {
			node.first = first_index;
			node.count = index_count;

			return;
		}
	}

	float split = primitives[indices[split_dimension][split_index]].get_center()[split_dimension];
	BVHPartitions::split_indices(primitives, indices, first_index, index_count, temp, split_dimension, split_index, split);

	node.count = (split_dimension + 1) << 30;

	int n_left  = split_index - first_index;
	int n_right = first_index + index_count - split_index;

	build_bvh(nodes[node.left    ], primitives, indices, nodes, node_index, first_index,          n_left,  sah, temp, max_primitives_in_leaf);
	build_bvh(nodes[node.left + 1], primitives, indices, nodes, node_index, first_index + n_left, n_right, sah, temp, max_primitives_in_leaf);
}

template<typename Primitive>
static BVH build_bvh_internal(const Primitive * primitives, int primitive_count, int max_primitives_in_leaf) {
	BVH bvh;
	bvh.nodes = new BVHNode[2 * primitive_count];

	// Construct index arrays for all three dimensions
	int * indices_x = new int[primitive_count];
	int * indices_y = new int[primitive_count];
	int * indices_z = new int[primitive_count];

	for (int i = 0; i < primitive_count; i++) {
		indices_x[i] = i;
		indices_y[i] = i;
		indices_z[i] = i;
	}

	std::sort(indices_x, indices_x + primitive_count, [&](int a, int b) { return primitives[a].get_center().x < primitives[b].get_center().x; });
	std::sort(indices_y, indices_y + primitive_count, [&](int a, int b) { return primitives[a].get_center().y < primitives[b].get_center().y; });
	std::sort(indices_z, indices_z + primitive_count, [&](int a, int b) { return primitives[a].get_center().z < primitives[b].get_center().z; });
		
	int * indices_3[3] = { indices_x, indices_y, indices_z };
		
	float * sah = new float[primitive_count];

	int * temp = new int[primitive_count];

	int node_index = 2;
	build_bvh(bvh.nodes[0], primitives, indices_3, bvh.nodes, node_index, 0, primitive_count, sah, temp, max_primitives_in_leaf);

	bvh.indices = indices_x;
	delete [] indices_y;
	delete [] indices_z;

	assert(node_index <= 2 * primitive_count);

	bvh.node_count  = node_index;
	bvh.index_count = primitive_count;

	delete [] temp;
	delete [] sah;

	return bvh;
}

BVH BVHBuilders::build_bvh(const Triangle * triangles, int triangle_count) {
	return build_bvh_internal(triangles, triangle_count, INT_MAX);
}

BVH BVHBuilders::build_bvh(const Mesh * meshes, int mesh_count) {
	return build_bvh_internal(meshes, mesh_count, 1);
}
