#pragma once
#include "Pathtracer/Triangle.h"

#include "Util/String.h"

namespace OBJLoader {
	bool load(const String & filename, Triangle *& triangles, int & triangle_count);
}
