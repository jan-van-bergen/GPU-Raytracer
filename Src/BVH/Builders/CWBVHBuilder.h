#pragma once
#include "BVH/BVH.h"

struct CWBVHBuilder {
	inline void init(BVH8 * cwbvh, const BVH2 & bvh) {
		this->cwbvh = cwbvh;

		cwbvh->indices.reserve(bvh.indices.size());
		cwbvh->nodes  .reserve(bvh.nodes  .size());
		cwbvh->nodes.emplace_back();
	}

	void build(const BVH2 & bvh);

private:
	BVH8 * cwbvh;

	struct Decision {
		enum struct Type : char {
			LEAF,
			INTERNAL,
			DISTRIBUTE
		} type;

		char distribute_left  = INVALID;
		char distribute_right = INVALID;

		float cost;
	};

	Array<Decision> decisions;

	int calculate_cost(int node_index, const Array<BVHNode2> & nodes);

	void get_children  (int node_index, const Array<BVHNode2> & nodes, int i, int & child_count, int children[8]);
	void order_children(int node_index, const Array<BVHNode2> & nodes, int children[8], int child_count);

	int count_primitives(int node_index, const Array<BVHNode2> & nodes, const Array<int> & indices_sbvh);

	void collapse(const Array<BVHNode2> & nodes_bvh, const Array<int> & indices_bvh, int node_index_cwbvh, int node_index_bvh);
};
