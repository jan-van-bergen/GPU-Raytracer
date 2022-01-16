#pragma once
#include "Pathtracer/Triangle.h"

#include "BVH/BVH.h"

#include "Core/Array.h"
#include "Core/OwnPtr.h"

struct MeshData {
	Array<Triangle> triangles;
	OwnPtr<BVH>     bvh;
};

struct MeshDataHandle { int handle = INVALID; };
