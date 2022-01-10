#pragma once
#include "Util/Array.h"
#include "Util/Parser.h"

struct Triangle;

namespace MitshairLoader {
	Array<Triangle> load(const String & filename, SourceLocation location_in_mitsuba_file, float radius);
}
