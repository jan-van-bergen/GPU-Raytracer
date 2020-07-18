#pragma once
#include <vector>

#include "Triangle.h"

#include "BVH.h"
#include "QBVH.h"
#include "CWBVH.h"

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	typedef BVH     BVHType;
	typedef BVHNode BVHNodeType;
#elif BVH_TYPE == BVH_QBVH
	typedef QBVH     BVHType;
	typedef QBVHNode BVHNodeType;
#elif BVH_TYPE == BVH_CWBVH
	typedef CWBVH     BVHType;
	typedef CWBVHNode BVHNodeType;
#endif

struct MeshData {
	int        triangle_count;
	Triangle * triangles;

	BVHType bvh;

	int material_offset;
	
	mutable unsigned gl_vbo;

	void init_gl(int reverse_indices[]) const;
	void render() const;

	static int load(const char * filename);

	inline static std::vector<const MeshData *> mesh_datas;
};
