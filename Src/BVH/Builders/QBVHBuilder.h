#pragma once
#include "BVH/BVH.h"

struct QBVHBuilder {
private:
	BVH4 * qbvh;

	void collapse(int node_index);

public:
	inline void init(BVH4 * qbvh, const BVH2 & bvh) {
		this->qbvh = qbvh;

		qbvh->indices = bvh.indices;
		qbvh->nodes.resize(bvh.nodes.size());
	}

	void build(const BVH2 & bvh);
};
