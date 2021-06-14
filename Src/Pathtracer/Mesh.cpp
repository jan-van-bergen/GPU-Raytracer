#include "Mesh.h"

#include "Pathtracer/Scene.h"

void Mesh::init(const char * name, int mesh_data_index, Scene & scene) {
	this->name = name;
	this->mesh_data_index = mesh_data_index;

	const MeshData * mesh_data = scene.mesh_datas[mesh_data_index];

	aabb_untransformed = AABB::create_empty();
	for (int i = 0; i < mesh_data->triangle_count; i++) {
		aabb_untransformed.expand(mesh_data->triangles[i].aabb);
	}
}

void Mesh::update() {
	// Update Transform
	transform_prev = transform;

	transform =
		Matrix4::create_translation(position) *
		Matrix4::create_rotation(rotation) *
		Matrix4::create_scale(scale);
	transform_inv =
		Matrix4::create_scale(1.0f / scale) *
		Matrix4::create_rotation(Quaternion::conjugate(rotation)) *
		Matrix4::create_translation(-position);

	// Update AABB from Transform
	aabb = AABB::transform(aabb_untransformed, transform);
	assert(aabb.is_valid());
}
