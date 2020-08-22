#pragma once
#include "MeshData.h"

struct Mesh {
	AABB aabb_untransformed;
	AABB aabb;

	int mesh_data_index;
	
	Vector3    position;
	Quaternion rotation;
	float      scale = 1.0f;

	Matrix4 transform;
	Matrix4 transform_inv;
	Matrix4 transform_prev;

	int   light_index = -1;
	float light_area = 0.0f;

	void init(int mesh_data_index);

	void update();

	inline Vector3 get_center() const { return aabb.get_center(); }
};
