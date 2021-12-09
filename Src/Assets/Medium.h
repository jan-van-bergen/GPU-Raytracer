#pragma once
#include "Math/Vector3.h"

#include "../CUDA_Source/Common.h"

struct Medium {
	const char * name;

	Vector3 scatter_coefficient;
    Vector3 negative_absorption;
};

struct MediumHandle { int handle = INVALID; };
