#pragma once
#include "MeshData.h"

struct Mesh {
	AABB aabb_untransformed;
	AABB aabb;

	int mesh_data_index;
	
	Vector3    position;
	Quaternion rotation;

	Matrix4 transform;
	Matrix4 transform_inv;
	Matrix4 transform_prev;

	void init(int mesh_data_index);

	void update();

	inline Vector3 get_center() const { return aabb.get_center(); }
};
