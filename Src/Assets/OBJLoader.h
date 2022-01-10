#pragma once
#include "Pathtracer/Triangle.h"

#include "Util/String.h"

namespace OBJLoader {
	Array<Triangle> load(const String & filename);
}
