#pragma once
#include "BVH.h"

struct Mesh {
	BVH bvh;

	int material_offset;

	static const Mesh * load(const char * filename);
};
