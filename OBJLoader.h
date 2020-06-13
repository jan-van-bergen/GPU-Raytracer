#pragma once

#include "Mesh.h"

namespace OBJLoader {
	void load_mtl(const char * filename, Mesh * mesh); // Only loads materials
	void load_obj(const char * filename, Mesh * mesh); // Loads geometry + materials
}
