#pragma once
#include "BVH/BVH.h"

struct CWBVHBuilder {
private:
	CWBVH * cwbvh;
	
	struct Decision {
		enum struct Type : char {
			LEAF,
			INTERNAL,
			DISTRIBUTE
		} type;

		char distribute_0 = -1;
		char distribute_1 = -1;
	};
	
	float    * cost;
	Decision * decisions;

	int calculate_cost(int node_index, const BVHNode nodes[]);

	void get_children  (int node_index, const BVHNode nodes[], int i, int & child_count, int children[8]);
	void order_children(int node_index, const BVHNode nodes[], int children[8], int child_count);

	int count_primitives(int node_index, const BVHNode nodes[], const int indices_sbvh[]);

	void collapse(const BVHNode nodes_sbvh[], const int indices_sbvh[], int node_index_wbvh, int node_index_sbvh);

public:
	inline void init(CWBVH * cwbvh, const BVH & bvh) {
		this->cwbvh = cwbvh;

		cost      = new float   [bvh.node_count * 7];
		decisions = new Decision[bvh.node_count * 7];

		cwbvh->index_count = bvh.index_count;
		cwbvh->indices     = bvh.indices;
		cwbvh->node_count  = bvh.node_count;
		cwbvh->nodes       = new CWBVHNode[cwbvh->node_count];
	}

	inline void free() {
		delete [] cost;
		delete [] decisions;
	}

	void build(const BVH & bvh);
};
