#pragma once
#include "Pathtracer/Triangle.h"

#include "BVH/BVH.h"

struct MeshData {
	int        triangle_count;
	Triangle * triangles;

	BVHType bvh;

	int material_offset;
	
	static int load(const char * filename, struct Scene & scene);
};
