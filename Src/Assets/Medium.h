#pragma once
#include "Math/Vector3.h"

#include "../CUDA_Source/Common.h"

struct Medium {
	const char * name;

	Vector3 sigma_a;
	Vector3 sigma_s;
	float   scale = 1.0f;

	float g = 0.0f;
};

struct MediumHandle { int handle = INVALID; };
