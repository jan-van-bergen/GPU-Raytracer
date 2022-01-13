#pragma once
#include "BVHConverter.h"

struct BVH4Converter final : BVHConverter {
	      BVH4 & bvh4;
	const BVH2 & bvh2;

	BVH4Converter(BVH4 & bvh4, const BVH2 & bvh2) : bvh4(bvh4), bvh2(bvh2) {
		bvh4.indices = bvh2.indices;
		bvh4.nodes.resize(bvh2.nodes.size());
	}

	void convert() override;

private:
	void collapse(int node_index);
};
