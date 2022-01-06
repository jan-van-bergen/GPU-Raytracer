#pragma once
#include "Pathtracer/Triangle.h"

#include "Util/StringView.h"

namespace PLYLoader {
	void load(const String & filename, Triangle *& triangles, int & triangle_count);
}
