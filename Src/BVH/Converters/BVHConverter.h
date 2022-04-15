#pragma once
#include "BVH/BVH.h"

// Converts a binary BVH (BVH2) into an n-ary BVH
struct BVHConverter {
	BVHConverter() = default;

	NON_COPYABLE(BVHConverter);
	NON_MOVEABLE(BVHConverter);

	virtual ~BVHConverter() = default;

	virtual void convert() = 0;
};

// Dummy converter, does nothing but copy
struct BVH2Converter final : BVHConverter {
	      BVH2 & bvh_to;
	const BVH2 & bvh_from;

	BVH2Converter(BVH2 & bvh_to, const BVH2 & bvh_from) : bvh_to(bvh_to), bvh_from(bvh_from) { }

	void convert() override {
		bvh_to = bvh_from;
	}
};
