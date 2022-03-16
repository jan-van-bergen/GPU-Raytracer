#pragma once
#include <cstdlib>

#include "Config.h"

#include "Core/String.h"
#include "Core/StringView.h"

#include "BVH/BVH.h"
#include "Renderer/MeshData.h"

namespace BVHLoader {
	inline constexpr const char * BVH_FILE_EXTENSION = ".bvh";
	inline constexpr int          BVH_FILETYPE_VERSION = 5;

	String get_bvh_filename(StringView filename, Allocator * allocator);

	bool try_to_load(const String & filename, const String & bvh_filename, MeshData * mesh_data, BVH2 * bvh);
	bool save(const String & bvh_filename, const MeshData & mesh_data, const BVH2 & bvh);
}
