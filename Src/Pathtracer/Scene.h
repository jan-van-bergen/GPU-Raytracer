#pragma once
#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

#include "Assets/AssetManager.h"

struct SceneConfig {
	Array<const char *> scenes;
	const char * sky = nullptr;
};

struct Scene {
	AssetManager asset_manager;

	Camera      camera;
	Array<Mesh> meshes;
	Sky         sky;

	bool has_diffuse    = false;
	bool has_dielectric = false;
	bool has_conductor  = false;
	bool has_lights     = false;

	void init(const SceneConfig & scene_config);

	Mesh & add_mesh(const char * name, MeshDataHandle mesh_data_handle, MaterialHandle material_handle = MaterialHandle::get_default());

	void check_materials();

	void update(float delta);
};
