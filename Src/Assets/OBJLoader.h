#pragma once
#include "Pathtracer/Triangle.h"

#include "Core/String.h"

namespace OBJLoader {
	Array<Triangle> load(const String & filename);
}
