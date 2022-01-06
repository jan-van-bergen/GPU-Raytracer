#pragma once
#include "Math/Vector3.h"

#include "Util/StringView.h"

struct Sky {
	int width;
	int height;
	Vector3 * data;

	void init(const String & file_name);
	void free();
};
