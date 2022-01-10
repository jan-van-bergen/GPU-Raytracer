#pragma once
#include <stdint.h>

#include "Util/Parser.h"

struct Triangle;
struct XMLNode;

struct Serialized {
	Array<Array<Triangle>> meshes;
};

namespace SerializedLoader {
	Serialized load(const String & filename, SourceLocation location_in_mitsuba_file);
}
