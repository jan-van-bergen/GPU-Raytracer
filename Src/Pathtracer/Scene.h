#pragma once
#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

#include "Assets/AssetManager.h"

struct Scene {
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

	void init(const SceneConfig & scene_config);

	Mesh & add_mesh(const char * name, MeshDataHandle mesh_data_handle, MaterialHandle material_handle = MaterialHandle::get_default());

	void calc_properties();

	void update(float delta);
};
