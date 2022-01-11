#pragma once
#include "Pathtracer/Triangle.h"

#include "BVH/BVH.h"

#include "Util/Array.h"

struct MeshData {
	Array<Triangle> triangles;
	BVH           * bvh;
};

struct MeshDataHandle { int handle = INVALID; };
