#pragma once
#include <cstdlib>

#include "Config.h"

#include "BVH/BVH.h"
#include "MeshData.h"

namespace BVHLoader {
	inline constexpr const char * BVH_FILE_EXTENSION = ".bvh";
	inline constexpr int          BVH_FILETYPE_VERSION = 4;

	bool try_to_load(const char * filename, MeshData & mesh_data, BVH & bvh);
	bool save       (const char * filename, MeshData & mesh_data, BVH & bvh);
}
