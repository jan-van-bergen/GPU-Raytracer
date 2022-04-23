#pragma once
#include "Math/Vector3.h"

#include "Core/String.h"

struct Sky {
	Array<Vector3> data;
	int            width;
	int            height;
	float          scale = 1.0f;

	void load(const String & file_name);
};
