#pragma once
#include "BVH.h"
#include "QBVH.h"
#include "CWBVH.h"

namespace BVHBuilders {
	void build_bvh (BVH & bvh); // SAH-based (object splits)
	void build_sbvh(BVH & bvh); // SAH-based (object + spatial splits, Stich et al. 2009)

	QBVH   qbvh_from_binary_bvh(const BVH & bvh);
	CWBVH cwbvh_from_binary_bvh(const BVH & bvh);
}
