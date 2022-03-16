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

	Handle<MeshData> mesh_data_handle;

	Vector3    position;
	Quaternion rotation;
	float      scale = 1.0f;

	Vector3 euler_angles; // For editor only

	Handle<Material> material_handle;

	Matrix4 transform;
	Matrix4 transform_inv;
	Matrix4 transform_prev;

	struct {
		float weight = 0.0f;

		int first_triangle_index;
		int triangle_count;
	} light;

	Mesh(String name, Handle<MeshData> mesh_data_handle, Handle<Material> material_handle);

	void calc_aabb(const Scene & scene);

	void update();

	bool has_identity_transform() const;

	inline Vector3 get_center() const { return aabb.get_center(); }
};
