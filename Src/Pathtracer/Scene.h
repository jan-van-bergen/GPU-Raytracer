#pragma once
#include <vector>

#include <mutex>
#include <thread>

#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

#include "Assets/AssetManager.h"

struct Scene {
	AssetManager asset_manager;

	Camera camera;
	std::vector<Mesh> meshes;
	Sky sky;
	
	bool has_diffuse    = false;
	bool has_dielectric = false;
	bool has_glossy     = false;
	bool has_lights     = false;

	void init(const char * scene_name, const char * sky_name);

	void check_materials();

	void update(float delta);
};
