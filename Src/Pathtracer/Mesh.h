#pragma once
#include "Assets/MeshData.h"

struct Scene;

struct Mesh {
	const char * name;

	AABB aabb_untransformed;
	AABB aabb;

	int mesh_data_id;
	
	Vector3    position;
	Quaternion rotation;
	float      scale = 1.0f;

	Vector3 euler_angles; // For editor only

	int material_id;

	Matrix4 transform;
	Matrix4 transform_inv;
	Matrix4 transform_prev;

	int   light_index = -1;
	float light_power = 0.0f;

	void init(const char * name, int mesh_data_id, Scene & scene);

	void update();

	inline Vector3 get_center() const { return aabb.get_center(); }
};
