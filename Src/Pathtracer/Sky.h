#pragma once
#include "Math/Vector3.h"

#include "Core/String.h"

struct Sky {
	int width;
	int height;
	Vector3 * data;

	void init(const String & file_name);
	void free();
};
