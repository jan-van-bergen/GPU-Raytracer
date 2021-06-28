#pragma once
#include <algorithm>

#include "BVH/BVH.h"
#include "BVHPartitions.h"

#include "Pathtracer/Mesh.h"

struct BVHBuilder {
private:
	BVH * bvh = nullptr;

	int * indices_x = nullptr;
	int * indices_y = nullptr;
	int * indices_z = nullptr;
		
	float * sah  = nullptr;
	int   * temp = nullptr;

	int max_primitives_in_leaf;

	template<typename Primitive>
	void build_bvh_recursive(BVHNode & node, const Primitive * primitives, const Vector3 * centers, int * indices[3], int & node_index, int first_index, int index_count) {
		node.aabb = BVHPartitions::calculate_bounds(primitives, indices[0], first_index, first_index + index_count);
		
		if (index_count == 1) {
			// Leaf Node, terminate recursion
			node.first = first_index;
			node.count = index_count;

			return;
		}

		int   split_dimension;
		float split_cost;
		int   split_index = BVHPartitions::partition_sah(primitives, indices, first_index, index_count, sah, split_dimension, split_cost);

#if !BVH_ENABLE_OPTIMIZATION && BVH_TYPE != BVH_CWBVH // BVH Optimizer and CWBVH both expect leaves with only a single primitive
		if (index_count <= max_primitives_in_leaf){
			// Check SAH termination condition
			float leaf_cost = node.aabb.surface_area() * SAH_COST_LEAF * float(index_count);
			float node_cost = node.aabb.surface_area() * SAH_COST_NODE + split_cost;

			if (leaf_cost < node_cost) {
				node.first = first_index;
				node.count = index_count;

				return;
			}
		}
#endif

		node.left = node_index;
		node_index += 2;
		
		float split = centers[indices[split_dimension][split_index]][split_dimension];
		BVHPartitions::split_indices(primitives, indices, first_index, index_count, temp, split_dimension, split_index, split);

		node.count = (split_dimension + 1) << 30;

		int num_left  = split_index - first_index;
		int num_right = first_index + index_count - split_index;

		build_bvh_recursive(bvh->nodes[node.left    ], primitives, centers, indices, node_index, first_index,            num_left);
		build_bvh_recursive(bvh->nodes[node.left + 1], primitives, centers, indices, node_index, first_index + num_left, num_right);
	}

	template<typename Primitive>
	inline void build_bvh_impl(const Primitive * primitives, int primitive_count) {
		Vector3 * centers = new Vector3[primitive_count];

		for (int i = 0; i < primitive_count; i++) {
			centers[i] = primitives[i].get_center();
		}

		std::sort(indices_x, indices_x + primitive_count, [centers](int a, int b) { return centers[a].x < centers[b].x; });
		std::sort(indices_y, indices_y + primitive_count, [centers](int a, int b) { return centers[a].y < centers[b].y; });
		std::sort(indices_z, indices_z + primitive_count, [centers](int a, int b) { return centers[a].z < centers[b].z; });
		
		int * indices[3] = { indices_x, indices_y, indices_z };
	
		int node_index = 2;
		build_bvh_recursive(bvh->nodes[0], primitives, centers, indices, node_index, 0, primitive_count);

		assert(node_index <= 2 * primitive_count);

		bvh->node_count  = node_index;
		bvh->index_count = primitive_count;

		delete [] centers;
	}

public:
	inline void init(BVH * bvh, int primitive_count, int max_primitives_in_leaf) {
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
		bvh->nodes   = new BVHNode[2 * primitive_count];
	}

	inline void free() {
		delete [] indices_y;
		delete [] indices_z;

		delete [] sah;
		delete [] temp;
	}
	
	inline void build(const Triangle * triangles, int triangle_count) {
		return build_bvh_impl(triangles, triangle_count);
	}

	inline void build(const Mesh * meshes, int mesh_count) {
		return build_bvh_impl(meshes, mesh_count);
	}
};
