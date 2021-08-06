#pragma once
#include <cstdlib>

#include "../CUDA_Source/Common.h"

#include "BVH/BVH.h"
#include "MeshData.h"

namespace BVHLoader {
	inline constexpr int UNDERLYING_BVH_TYPE    = BVH_TYPE == BVH_SBVH ? BVH_SBVH : BVH_BVH; // All BVH use standard BVH as underlying type, only SBVH uses SBVH
	inline constexpr int MAX_PRIMITIVES_IN_LEAF = BVH_TYPE == BVH_CWBVH || BVH_ENABLE_OPTIMIZATION ? 1 : INT_MAX; // CWBVH and BVH optimization require 1 primitive per leaf Node, the others have no upper limits

	inline constexpr const char * BVH_FILE_EXTENSION = ".bvh";
	inline constexpr int          BVH_FILETYPE_VERSION = 3;

	bool try_to_load(const char * filename, MeshData & mesh_data, BVH & bvh);
	bool save       (const char * filename, MeshData & mesh_data, BVH & bvh);
}
