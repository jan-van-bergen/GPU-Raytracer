#pragma once
#include "Core/BitArray.h"

#include "BVH/BVH.h"

struct Triangle;
struct Mesh;

struct SAHBuilder {
	BVH2 & bvh;

	Array<int> indices_x;
	Array<int> indices_y;
	Array<int> indices_z;

	Array<char> scratch; // Used to store intermediate SAH results and reorder indices
	BitArray indices_going_left;

	SAHBuilder(BVH2 & bvh, size_t primitive_count) :
		bvh(bvh),
		indices_x(primitive_count),
		indices_y(primitive_count),
		indices_z(primitive_count),
		scratch(primitive_count * Math::max(sizeof(float), sizeof(int))),
		indices_going_left(primitive_count)
	{
		for (int i = 0; i < primitive_count; i++) {
			indices_x[i] = i;
			indices_y[i] = i;
			indices_z[i] = i;
		}

		bvh.nodes.reserve(2 * primitive_count);
	}

	void build(const Array<Triangle> & triangles);
	void build(const Array<Mesh>     & meshes);
};
