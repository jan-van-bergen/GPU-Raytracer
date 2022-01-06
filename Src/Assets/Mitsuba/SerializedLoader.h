#pragma once
#include <stdint.h>

#include "Util/Parser.h"

struct Triangle;
struct XMLNode;

struct Serialized {
	uint32_t num_meshes;

	int       * triangle_count;
	Triangle ** triangles;
};

namespace SerializedLoader {
	Serialized load(const String & filename, SourceLocation location_in_mitsuba_file);
}
