#pragma once
#include "Math/Vector2.h"

#include "Array.h"

struct PMJ {
	Vector2 * samples;

	void init();
	void free();
};
