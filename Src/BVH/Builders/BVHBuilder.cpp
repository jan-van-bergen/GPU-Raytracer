#include "BVHBuilder.h"

#include "BVH/BVH.h"
#include "BVHPartitions.h"

#include "Pathtracer/Mesh.h"

template<typename Primitive>
static void build_bvh_recursive(BVHBuilder & builder, BVHNode2 & node, const Primitive * primitives, const Vector3 * centers, int * indices[3], int & node_index, int first_index, int index_count) {
	node.aabb = BVHPartitions::calculate_bounds(primitives, indices[0], first_index, first_index + index_count);

	if (index_count == 1) {
		// Leaf Node, terminate recursion
		node.first = first_index;
		node.count = index_count;

		return;
	}

	int   split_dimension;
	float split_cost;
	int   split_index = BVHPartitions::partition_sah(primitives, indices, first_index, index_count, builder.sah, split_dimension, split_cost);

	if (index_count <= builder.max_primitives_in_leaf) {
		// Check SAH termination condition
		float leaf_cost = node.aabb.surface_area() * config.sah_cost_leaf * float(index_count);
		float node_cost = node.aabb.surface_area() * config.sah_cost_node + split_cost;

		if (leaf_cost < node_cost) {
			node.first = first_index;
			node.count = index_count;

			return;
		}
	}

	node.left = node_index;
	node_index += 2;

	float split = centers[indices[split_dimension][split_index]][split_dimension];
	BVHPartitions::split_indices(primitives, indices, first_index, index_count, builder.temp, split_dimension, split_index, split);

	node.count = 0;
	node.axis  = split_dimension;

	int num_left  = split_index - first_index;
	int num_right = first_index + index_count - split_index;

	build_bvh_recursive(builder, builder.bvh->nodes_2[node.left    ], primitives, centers, indices, node_index, first_index,            num_left);
	build_bvh_recursive(builder, builder.bvh->nodes_2[node.left + 1], primitives, centers, indices, node_index, first_index + num_left, num_right);
}

template<typename Primitive>
static void build_bvh_impl(BVHBuilder & builder, const Primitive * primitives, int primitive_count) {
	Vector3 * centers = new Vector3[primitive_count];

	for (int i = 0; i < primitive_count; i++) {
		centers[i] = primitives[i].get_center();
	}

	Util::quick_sort(builder.indices_x, builder.indices_x + primitive_count, [centers](int a, int b) { return centers[a].x < centers[b].x; });
	Util::quick_sort(builder.indices_y, builder.indices_y + primitive_count, [centers](int a, int b) { return centers[a].y < centers[b].y; });
	Util::quick_sort(builder.indices_z, builder.indices_z + primitive_count, [centers](int a, int b) { return centers[a].z < centers[b].z; });

	int * indices[3] = {
		builder.indices_x,
		builder.indices_y,
		builder.indices_z
	};

	int node_index = 2;
	build_bvh_recursive(builder, builder.bvh->nodes_2[0], primitives, centers, indices, node_index, 0, primitive_count);

	assert(node_index <= 2 * primitive_count);

	builder.bvh->node_count  = node_index;
	builder.bvh->index_count = primitive_count;

	delete [] centers;
}

void BVHBuilder::init(BVH * bvh, int primitive_count, int max_primitives_in_leaf) {
	this->bvh = bvh;
	this->max_primitives_in_leaf = max_primitives_in_leaf;

	indices_x = new int[primitive_count];
	indices_y = new int[primitive_count];
	indices_z = new int[primitive_count];

	for (int i = 0; i < primitive_count; i++) {
		indices_x[i] = i;
		indices_y[i] = i;
		indices_z[i] = i;
	}

	sah  = new float[primitive_count];
	temp = new int  [primitive_count];

	bvh->indices = indices_x;
	bvh->nodes_2 = new BVHNode2[2 * primitive_count];
}

void BVHBuilder::free() {
	delete [] indices_y;
	delete [] indices_z;

	delete [] sah;
	delete [] temp;
}

void BVHBuilder::build(const Triangle * triangles, int triangle_count) {
	return build_bvh_impl(*this, triangles, triangle_count);
}

void BVHBuilder::build(const Mesh * meshes, int mesh_count) {
	return build_bvh_impl(*this, meshes, mesh_count);
}
