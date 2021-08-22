#pragma once
#include "Pathtracer/Triangle.h"

namespace PLYLoader {
	void load(const char * filename, Triangle *& triangles, int & triangle_count);
}
