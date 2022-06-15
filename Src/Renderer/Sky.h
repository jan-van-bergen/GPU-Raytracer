#pragma once
#include "Math/Vector4.h"

#include "Core/String.h"

struct Sky {
	Array<Vector4> data;
	int            width;
	int            height;
	float          scale = 1.0f;

	void load(const String & file_name);
};
