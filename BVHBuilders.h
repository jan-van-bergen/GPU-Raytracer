#pragma once
#include "MeshData.h"

// Forward declarations
struct BVH;
struct BVHNode;

struct QBVH;

struct CWBVH;

namespace BVHBuilders {
	void build_bvh (BVHNode & node, const Triangle * triangles, int * indices[3], BVHNode nodes[], int & node_index, int first_index, int index_count, float * sah, int * temp);
	int  build_sbvh(BVHNode & node, const Triangle * triangles, int * indices[3], BVHNode nodes[], int & node_index, int first_index, int index_count, float * sah, int * temp[2], float inv_root_surface_area, AABB node_aabb);

	QBVH   qbvh_from_binary_bvh(const BVH & bvh);
	CWBVH cwbvh_from_binary_bvh(const BVH & bvh);
}
