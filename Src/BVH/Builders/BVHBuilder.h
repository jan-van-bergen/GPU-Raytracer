#pragma once
#include "BVH/BVH.h"
#include "Util/BitArray.h"

struct Triangle;
struct Mesh;

struct BVHBuilder {
	BVH * bvh = nullptr;

	int * indices_x = nullptr;
	int * indices_y = nullptr;
	int * indices_z = nullptr;

	char * scratch = nullptr; // Used to store intermediate SAH results and reorder indices
	BitArray indices_going_left;

	void init(BVH * bvh, int primitive_count);
	void free();

	void build(const Array<Triangle> & triangles);
	void build(const Array<Mesh>     & meshes);
};
