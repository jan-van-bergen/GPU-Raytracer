#pragma once
#include "Renderer/Triangle.h"

#include "Core/String.h"

namespace OBJLoader {
	Array<Triangle> load(const String & filename, Allocator * allocator);
}
