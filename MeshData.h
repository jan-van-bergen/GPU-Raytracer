#pragma once
#include "Vector2.h"
#include "Vector3.h"

#include "Material.h"

struct Triangle {
	Vector3 position0;
	Vector3 position1;
	Vector3 position2;

	Vector3 normal0;
	Vector3 normal1;
	Vector3 normal2;

	alignas(8) Vector2 tex_coord0;
	alignas(8) Vector2 tex_coord1;
	alignas(8) Vector2 tex_coord2;

	int material_id;
};

struct MeshData {
	int        triangle_count;
	Triangle * triangles;

	int        material_count;
	Material * materials;

	static const MeshData * load(const char * file_path);
};
