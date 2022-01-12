#pragma once
#include "Pathtracer/Triangle.h"

#include "BVH/BVH.h"

#include "Util/Array.h"
#include "Util/OwnPtr.h"

struct MeshData {
	Array<Triangle> triangles;
	OwnPtr<BVH>     bvh;
};

struct MeshDataHandle { int handle = INVALID; };
