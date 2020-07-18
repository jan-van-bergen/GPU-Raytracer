#pragma once
#include "Mesh.h"

namespace BVHBuilders {
	BVH build_bvh(const Triangle * triangles, int triangle_count); // SAH-based object splits (for BLAS)
	BVH build_bvh(const Mesh     * meshes,    int mesh_count);     // SAH-based object splits (for TLAS)

	BVH build_sbvh(const Triangle * triangles, int triangle_count); // SAH-based object + spatial splits, Stich et al. 2009 (Triangles only)

	QBVH   qbvh_from_binary_bvh(const BVH & bvh);
	CWBVH cwbvh_from_binary_bvh(const BVH & bvh);
}
