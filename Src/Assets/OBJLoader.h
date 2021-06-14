#pragma once

#include "Assets/MeshData.h"

struct Scene;

namespace OBJLoader {
	void load_mtl(const char * filename, MeshData * mesh_data, Scene & scene); // Only loads materials
	void load_obj(const char * filename, MeshData * mesh_data, Scene & scene); // Loads geometry + materials
}
