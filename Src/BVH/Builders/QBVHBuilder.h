#pragma once
#include "BVH/BVH.h"

struct QBVHBuilder {
private:
	QBVH * qbvh;

	void collapse(int node_index);

public:
	inline void init(QBVH * qbvh, const BVH & bvh) {
		this->qbvh = qbvh;

		qbvh->index_count = bvh.index_count;
		qbvh->indices     = bvh.indices;

		qbvh->node_count = bvh.node_count;
		qbvh->nodes      = new QBVHNode[bvh.node_count];

	}

	void build(const BVH & bvh);
};
