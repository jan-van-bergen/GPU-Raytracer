#pragma once
#include "Math/Vector3.h"

#include "Core/String.h"

struct Sky {
	int width;
	int height;
	Array<Vector3> data;

	void load(const String & file_name);
};
