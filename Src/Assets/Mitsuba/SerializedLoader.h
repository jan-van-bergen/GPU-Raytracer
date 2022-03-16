#pragma once
#include <stdint.h>

#include "Core/Parser.h"

struct Triangle;

namespace SerializedLoader {
	Array<Triangle> load(const String & filename, Allocator * allocator, SourceLocation location_in_mitsuba_file, int shape_index);
}
