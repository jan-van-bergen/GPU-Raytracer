#pragma once
#include "BVH.h"

namespace BVHCollapser {
	// Collapses the given BVH based on the SAH cost
	// For each internal node we check whether it is cheaper to collapse
	// its entire subtree into a single leaf node based on the SAH cost
	void collapse(BVH2 & bvh);
}
