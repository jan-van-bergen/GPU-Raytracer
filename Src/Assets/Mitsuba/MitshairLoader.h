#pragma once
#include "Core/Array.h"
#include "Core/Parser.h"

struct Triangle;

namespace MitshairLoader {
	Array<Triangle> load(const String & filename, SourceLocation location_in_mitsuba_file, float radius);
}
