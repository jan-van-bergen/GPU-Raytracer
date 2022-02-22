#pragma once
#include "Core/Array.h"
#include "Core/Parser.h"

struct Triangle;

namespace MitshairLoader {
	Array<Triangle> load(const String & filename, Allocator * allocator, SourceLocation location_in_mitsuba_file, float radius);
}
