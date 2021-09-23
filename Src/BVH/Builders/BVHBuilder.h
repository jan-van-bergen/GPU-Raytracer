#pragma once
#include "BVH/BVH.h"

struct Triangle;
struct Mesh;

struct BVHBuilder {
	BVH * bvh = nullptr;

	Vector3 * centers = nullptr;

	int * indices_x = nullptr;
	int * indices_y = nullptr;
	int * indices_z = nullptr;

	float * sah  = nullptr;
	int   * temp = nullptr;

	int max_primitives_in_leaf;

	void init(BVH * bvh, int primitive_count, int max_primitives_in_leaf);
	void free();

	void build(const Triangle * triangles, int triangle_count);
	void build(const Mesh     * meshes,    int mesh_count);
};
