#pragma once
#include "MeshData.h"

// Forward declarations
struct BVH;
struct BVHNode;

struct QBVH;

struct CWBVH;

namespace BVHBuilders {
	BVH  bvh(const char * filename, const MeshData * mesh); // SAH-based (object splits)
	BVH sbvh(const char * filename, const MeshData * mesh); // SAH-based (object + spatial splits, Stich et al. 2009)

	QBVH   qbvh_from_binary_bvh(const BVH & bvh);
	CWBVH cwbvh_from_binary_bvh(const BVH & bvh);
}
