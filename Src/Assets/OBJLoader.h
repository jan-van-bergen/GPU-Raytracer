#pragma once

#include "Assets/MeshData.h"

namespace OBJLoader {
	void load_mtl(const char * filename, MeshData * mesh_data); // Only loads materials
	void load_obj(const char * filename, MeshData * mesh_data); // Loads geometry + materials
}
