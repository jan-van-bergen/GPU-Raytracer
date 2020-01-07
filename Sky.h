#pragma once
#include "Texture.h"

#include "Vector3.h"

struct Sky {
	int size;
	Vector3 * data;

	void init(const char * file_name);
};
