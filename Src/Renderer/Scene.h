#pragma once
#include "Core/Allocators/LinearAllocator.h"

#include "Assets/AssetManager.h"

#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

struct Scene {
	Allocator * allocator = nullptr;

	AssetManager asset_manager;

	Camera      camera;
	Array<Mesh> meshes;
	Sky         sky;

	bool has_diffuse    = false;
	bool has_plastic    = false;
	bool has_dielectric = false;
	bool has_conductor  = false;
	bool has_lights     = false;

	int triangle_count = 0;

	Scene(Allocator * allocator);

	Mesh & add_mesh(String name, Handle<MeshData> mesh_data_handle, Handle<Material> material_handle = Handle<Material>::get_default());

	void calc_properties();

	void update(float delta);
};
