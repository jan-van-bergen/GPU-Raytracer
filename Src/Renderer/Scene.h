#pragma once
#include "Core/Allocators/LinearAllocator.h"

#include "Assets/AssetManager.h"

#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

struct Scene {
	AssetManager asset_manager;

	Allocator * allocator = nullptr;

	Camera      camera;
	Array<Mesh> meshes;
	Sky         sky;

	bool has_diffuse    = false;
	bool has_plastic    = false;
	bool has_dielectric = false;
	bool has_conductor  = false;
	bool has_lights     = false;

	Scene(Allocator * allocator);

	Mesh & add_mesh(String name, MeshDataHandle mesh_data_handle, MaterialHandle material_handle = MaterialHandle::get_default());

	void check_materials();

	void update(float delta);
};
