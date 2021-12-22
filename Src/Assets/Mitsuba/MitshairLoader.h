#pragma once
#include "Util/Parser.h"

struct Triangle;

namespace MitshairLoader {
	void load(const char * filename, SourceLocation location_in_mitsuba_file, Triangle *& triangles, int & triangle_count, float radius);
}
