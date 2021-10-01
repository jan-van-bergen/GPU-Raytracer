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

	void init(BVH * bvh, int primitive_count);
	void free();

	void build(const Triangle * triangles, int triangle_count);
	void build(const Mesh     * meshes,    int mesh_count);
};
