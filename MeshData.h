#pragma once
#include "Vector2.h"
#include "Vector3.h"

#include "AABB.h"

#include "Material.h"

struct Triangle {
	AABB aabb;

	Vector3 position_0;
	Vector3 position_1;
	Vector3 position_2;

	Vector3 normal_0;
	Vector3 normal_1;
	Vector3 normal_2;

	Vector2 tex_coord_0;
	Vector2 tex_coord_1;
	Vector2 tex_coord_2;

	int material_id = -1;

	inline Vector3 get_center() const {
		return (position_0 + position_1 + position_2) / 3.0f;
	}
};

struct MeshData {
	int        triangle_count;
	Triangle * triangles;

	int        material_count;
	Material * materials;

	static const MeshData * load(const char * file_path);
};
