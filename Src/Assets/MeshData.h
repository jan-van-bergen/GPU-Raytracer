#pragma once
#include "Pathtracer/Triangle.h"

#include "BVH/BVH.h"

struct MeshData {
	int        triangle_count;
	Triangle * triangles;

	BVHType bvh;

	int material_offset;
	
	void init_bvh(const BVH & bvh);

	static int load(const char * filename, struct Scene & scene);
};
