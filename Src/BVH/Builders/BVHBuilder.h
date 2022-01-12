#pragma once
#include "BVH/BVH.h"
#include "Util/BitArray.h"

struct Triangle;
struct Mesh;

struct BVHBuilder {
	BVH2 * bvh = nullptr;

	Array<int> indices_x;
	Array<int> indices_y;
	Array<int> indices_z;

	Array<char> scratch; // Used to store intermediate SAH results and reorder indices
	BitArray indices_going_left;

	void init(BVH2 * bvh, int primitive_count);
	void free();

	void build(const Array<Triangle> & triangles);
	void build(const Array<Mesh>     & meshes);
};
