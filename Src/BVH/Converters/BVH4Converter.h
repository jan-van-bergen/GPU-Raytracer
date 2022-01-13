#pragma once
#include "BVHConverter.h"

struct BVH4Converter final : BVHConverter {
	      BVH4 & qbvh;
	const BVH2 & bvh;

	BVH4Converter(BVH4 & qbvh, const BVH2 & bvh) : qbvh(qbvh), bvh(bvh) {
		qbvh.indices = bvh.indices;
		qbvh.nodes.resize(bvh.nodes.size());
	}

	void convert() override;

private:
	void collapse(int node_index);
};
