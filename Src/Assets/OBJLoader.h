#pragma once
#include "Pathtracer/Triangle.h"

namespace OBJLoader {
	bool load(const char * filename, Triangle *& triangles, int & triangle_count);
}
