#pragma once
#include "BVH/BVH.h"

struct CWBVHBuilder {
private:
	BVH * cwbvh;

	struct Decision {
		enum struct Type : char {
			LEAF,
			INTERNAL,
			DISTRIBUTE
		} type;

		char distribute_left  = INVALID;
		char distribute_right = INVALID;
	};

	float    * cost;
	Decision * decisions;

	int calculate_cost(int node_index, const BVHNode2 nodes[]);

	void get_children  (int node_index, const BVHNode2 nodes[], int i, int & child_count, int children[8]);
	void order_children(int node_index, const BVHNode2 nodes[], int children[8], int child_count);

	int count_primitives(int node_index, const BVHNode2 nodes[], const int indices_sbvh[]);

	void collapse(const BVHNode2 nodes_bvh[], const int indices_bvh[], int node_index_cwbvh, int node_index_bvh);

public:
	inline void init(BVH * cwbvh, const BVH & bvh) {
		this->cwbvh = cwbvh;

		cost      = new float   [bvh.node_count * 7];
		decisions = new Decision[bvh.node_count * 7];

		cwbvh->index_count = 0;
		cwbvh->indices     = new int[bvh.index_count];
		cwbvh->node_count  = 1;
		cwbvh->nodes_8     = new BVHNode8[bvh.node_count];
	}

	inline void free() {
		delete [] cost;
		delete [] decisions;
	}

	void build(const BVH & bvh);
};
