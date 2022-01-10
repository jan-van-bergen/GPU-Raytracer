#pragma once
#include "Pathtracer/Triangle.h"

#include "Util/String.h"

namespace PLYLoader {
	Array<Triangle> load(const String & filename);
}
