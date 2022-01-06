#pragma once
#include "Pathtracer/Triangle.h"

#include "Util/String.h"

namespace PLYLoader {
	void load(const String & filename, Triangle *& triangles, int & triangle_count);
}
