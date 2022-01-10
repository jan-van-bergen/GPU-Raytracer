#pragma once
#include "Pathtracer/Triangle.h"

#include "BVH/BVH.h"

#include "Util/Array.h"

struct MeshData {
	Array<Triangle> triangles;

	BVH bvh;

	void init_bvh(BVH & bvh);
};

struct MeshDataHandle { int handle = INVALID; };
