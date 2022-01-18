#pragma once
#include "BVHConverter.h"

struct BVH8Converter final : BVHConverter {
	      BVH8 & bvh8;
	const BVH2 & bvh2;

	BVH8Converter(BVH8 & bvh8, const BVH2 & bvh2) : bvh8(bvh8), bvh2(bvh2) {
		bvh8.indices.reserve(bvh2.indices.size());
		bvh8.nodes  .reserve(bvh2.nodes  .size());
	}

	void convert() override;

private:
	struct Decision {
		enum struct Type : char {
			LEAF,
			INTERNAL,
			DISTRIBUTE
		} type;

		char distribute_left;
		char distribute_right;

		float cost;
	};

	Array<Decision> decisions;

	int calculate_cost(int node_index, const Array<BVHNode2> & nodes);

	void get_children  (int node_index, const Array<BVHNode2> & nodes, int children[8], int & child_count, int i);
	void order_children(int node_index, const Array<BVHNode2> & nodes, int children[8], int   child_count);

	int count_primitives(int node_index, const Array<BVHNode2> & nodes, const Array<int> & indices_sbvh);

	void collapse(const Array<BVHNode2> & nodes_bvh, const Array<int> & indices_bvh, int node_index_bvh8, int node_index_bvh2);
};
