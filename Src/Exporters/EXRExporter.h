#pragma once
#include "Core/Array.h"
#include "Core/String.h"

#include "Math/Vector3.h"

namespace EXRExporter {
	void save(const String & filename, int pitch, int width, int height, const Array<Vector3> & data);
}
