#pragma once
#include "Pathtracer/Triangle.h"

#include "BVH/BVH.h"

#include "../CUDA_Source/Common.h"

struct MeshData {
	int        triangle_count;
	Triangle * triangles;

	BVHType bvh;

	void init_bvh(const BVH & bvh);
};

struct MeshDataHandle { int handle = INVALID; };
