#pragma once
#include "Assets/MeshData.h"

struct Mesh {
	const char * name;

	AABB aabb_untransformed;
	AABB aabb;

	int mesh_data_index;
	
	Vector3    position;
	Quaternion rotation;
	float      scale = 1.0f;

	Vector3 euler_angles; // For editor only

	Matrix4 transform;
	Matrix4 transform_inv;
	Matrix4 transform_prev;

	int   light_index = -1;
	float light_area = 0.0f;

	void init(const char * name, int mesh_data_index);

	void update();

	inline Vector3 get_center() const { return aabb.get_center(); }
};
