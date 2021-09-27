#pragma once
#include <cstdlib>

#include "Config.h"

#include "BVH/BVH.h"
#include "MeshData.h"

namespace BVHLoader {
	inline constexpr const char * BVH_FILE_EXTENSION = ".bvh";
	inline constexpr int          BVH_FILETYPE_VERSION = 5;

	const char * get_bvh_filename(const char * filename);

	bool try_to_load(const char * filename, const char * bvh_filename, MeshData & mesh_data, BVH & bvh);
	bool save(const char * bvh_filename, const MeshData & mesh_data, const BVH & bvh);
}
