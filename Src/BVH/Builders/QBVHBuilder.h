#pragma once
#include "BVH/BVH.h"

struct QBVHBuilder {
private:
	BVH * qbvh;

	void collapse(int node_index);

public:
	inline void init(BVH * qbvh, const BVH & bvh) {
		this->qbvh = qbvh;

		qbvh->index_count = bvh.index_count;
		qbvh->indices     = bvh.indices; // Indices array can be reused

		qbvh->node_count = bvh.node_count;
		qbvh->nodes_4    = new BVHNode4[bvh.node_count];
	}

	void build(const BVH & bvh);
};
