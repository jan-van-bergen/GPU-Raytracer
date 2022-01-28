#pragma once
#include "Renderer/Triangle.h"

#include "Core/Array.h"
#include "Core/String.h"

namespace PLYLoader {
	Array<Triangle> load(const String & filename);
}
