#include "BVHBuilder.h"

#include <new>

#include "BVH/BVH.h"
#include "BVHPartitions.h"

#include "Pathtracer/Mesh.h"

void BVHBuilder::init(BVH * bvh, int primitive_count) {
	this->bvh = bvh;

	indices_x = new int[primitive_count];
	indices_y = new int[primitive_count];
	indices_z = new int[primitive_count];

	for (int i = 0; i < primitive_count; i++) {
		indices_x[i] = i;
		indices_y[i] = i;
		indices_z[i] = i;
	}

	scratch = new char[primitive_count * Math::max(sizeof(float), sizeof(int))];

	indices_going_left.init(primitive_count);

	bvh->indices  = indices_x;
	bvh->nodes._2 = new BVHNode2[2 * primitive_count];
}

void BVHBuilder::free() {
	delete [] indices_y;
	delete [] indices_z;

	delete [] scratch;

	indices_going_left.free();
}

template<typename Primitive>
static void build_bvh_recursive(BVHBuilder & builder, BVHNode2 & node, const Array<Primitive> & primitives, int * indices[3], int & node_index, int first_index, int index_count) {
	if (index_count == 1) {
		// Leaf Node, terminate recursion
		// We do not terminate based on the SAH termination criterion, so that the
		// BVHs that are cached to disk have a standard layout (1 triangle per leaf node)
		// If desired these trees can be collapsed based on the SAH cost using BVHCollapser::collapse
		node.first = first_index;
		node.count = index_count;

		return;
	}

	ObjectSplit split = BVHPartitions::partition_sah(primitives, indices, first_index, index_count, new(builder.scratch) float[index_count]);

	for (int i = first_index; i < split.index;               i++) builder.indices_going_left[indices[split.dimension][i]] = true;
	for (int i = split.index; i < first_index + index_count; i++) builder.indices_going_left[indices[split.dimension][i]] = false;

	for (int dim = 0; dim < 3; dim++) {
		if (dim == split.dimension) continue;

		int left  = 0;
		int right = split.index - first_index;
		int * temp = new(builder.scratch) int[index_count];

		for (int i = first_index; i < first_index + index_count; i++) {
			int index = indices[dim][i];

			bool goes_left = builder.indices_going_left[index];
			if (goes_left) {
				temp[left++] = index;
			} else {
				temp[right++] = index;
			}
		}

		ASSERT(left  == split.index - first_index);
		ASSERT(right == index_count);
		memcpy(indices[dim] + first_index, temp, index_count * sizeof(int));
	}

	node.left = node_index;
	node.count = 0;
	node.axis  = split.dimension;

	builder.bvh->nodes._2[node.left    ].aabb = split.aabb_left;
	builder.bvh->nodes._2[node.left + 1].aabb = split.aabb_right;

	node_index += 2;

	int num_left  = split.index - first_index;
	int num_right = first_index + index_count - split.index;

	build_bvh_recursive(builder, builder.bvh->nodes._2[node.left    ], primitives, indices, node_index, first_index,            num_left);
	build_bvh_recursive(builder, builder.bvh->nodes._2[node.left + 1], primitives, indices, node_index, first_index + num_left, num_right);
}

template<typename Primitive>
static void build_bvh_impl(BVHBuilder & builder, const Array<Primitive> & primitives) {
	AABB root_aabb = AABB::create_empty();
	for (size_t i = 0; i < primitives.size(); i++) {
		root_aabb.expand(primitives[i].aabb);
	}
	builder.bvh->nodes._2[0].aabb = root_aabb;

	Util::quick_sort(builder.indices_x, builder.indices_x + primitives.size(), [&primitives](int a, int b) { return primitives[a].get_center().x < primitives[b].get_center().x; });
	Util::quick_sort(builder.indices_y, builder.indices_y + primitives.size(), [&primitives](int a, int b) { return primitives[a].get_center().y < primitives[b].get_center().y; });
	Util::quick_sort(builder.indices_z, builder.indices_z + primitives.size(), [&primitives](int a, int b) { return primitives[a].get_center().z < primitives[b].get_center().z; });

	int * indices[3] = {
		builder.indices_x,
		builder.indices_y,
		builder.indices_z
	};

	int node_index = 2;
	build_bvh_recursive(builder, builder.bvh->nodes._2[0], primitives, indices, node_index, 0, primitives.size());

	ASSERT(node_index <= 2 * primitives.size());

	builder.bvh->node_count  = node_index;
	builder.bvh->index_count = primitives.size();
}

void BVHBuilder::build(const Array<Triangle> & triangles) {
	return build_bvh_impl(*this, triangles);
}

void BVHBuilder::build(const Array<Mesh> & meshes) {
	return build_bvh_impl(*this, meshes);
}
