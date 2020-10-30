#pragma once
#include <vector>

#include "Triangle.h"

#include "BVH.h"

struct MeshData {
	int        triangle_count;
	Triangle * triangles;

	BVHType bvh;

	int material_offset;
	
	mutable unsigned gl_vao;
	mutable unsigned gl_vbo;

	void gl_init(int reverse_indices[]) const;
	void gl_render() const;

	static int load(const char * filename);

	inline static std::vector<const MeshData *> mesh_datas;
};
