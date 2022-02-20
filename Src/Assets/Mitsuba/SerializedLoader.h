#pragma once
#include <stdint.h>

#include "Core/Parser.h"

struct Triangle;
struct XMLNode;

struct Serialized {
	Array<Array<Triangle>> meshes;
};

namespace SerializedLoader {
	Serialized load(const String & filename, Allocator * allocator, SourceLocation location_in_mitsuba_file);
}
