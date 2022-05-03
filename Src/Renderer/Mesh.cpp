#include "Mesh.h"

#include "Renderer/Scene.h"

Mesh::Mesh(String name, Handle<MeshData> mesh_data_handle, Handle<Material> material_handle) : name(std::move(name)), mesh_data_handle(mesh_data_handle), material_handle(material_handle) { }

void Mesh::calc_aabb(const Scene & scene) {
	const MeshData & mesh_data = scene.asset_manager.get_mesh_data(mesh_data_handle);

	aabb_untransformed = AABB::create_empty();
	for (int i = 0; i < mesh_data.triangles.size(); i++) {
		aabb_untransformed.expand(mesh_data.triangles[i].get_aabb());
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
	aabb.fix_if_needed();
	ASSERT(aabb.is_valid());
}

bool Mesh::has_identity_transform() const {
	constexpr float epsilon = 1e-6f;
	return
		Math::approx_equal(scale, 1.0f, epsilon) &&
		Math::approx_equal(position.x, 0.0f, epsilon) &&
		Math::approx_equal(position.y, 0.0f, epsilon) &&
		Math::approx_equal(position.z, 0.0f, epsilon) &&
		Math::approx_equal(rotation.x, 0.0f, epsilon) &&
		Math::approx_equal(rotation.y, 0.0f, epsilon) &&
		Math::approx_equal(rotation.z, 0.0f, epsilon) &&
		(Math::approx_equal(rotation.w, 1.0f, epsilon) || Math::approx_equal(rotation.w, -1.0f, epsilon)); // Due to double cover rotation may be (0,0,0,1) or (0,0,0,-1)
}
