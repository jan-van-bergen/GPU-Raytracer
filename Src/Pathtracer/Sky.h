#pragma once
#include "Math/Vector3.h"

struct Sky {
	int width;
	int height;
	Vector3 * data;

	void init(const char * file_name);
	void free();
};
