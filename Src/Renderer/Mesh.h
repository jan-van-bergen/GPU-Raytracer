#pragma once
#include "Math/Quaternion.h"
#include "Math/Matrix4.h"

#include "MeshData.h"
#include "Material.h"

struct Scene;

struct Mesh {
	String name;

	AABB aabb_untransformed;
	AABB aabb;

	MeshDataHandle mesh_data_handle;

	Vector3    position;
	Quaternion rotation;
	float      scale = 1.0f;

	Vector3 euler_angles; // For editor only

	MaterialHandle material_handle;

	Matrix4 transform;
	Matrix4 transform_inv;
	Matrix4 transform_prev;

	struct {
		float weight = 0.0f;

		int first_triangle_index;
		int triangle_count;
	} light;

	Mesh(String name, MeshDataHandle mesh_data_handle, MaterialHandle material_handle, const Scene & scene);

	void update();

	bool has_identity_transform() const;

	inline Vector3 get_center() const { return aabb.get_center(); }
};
