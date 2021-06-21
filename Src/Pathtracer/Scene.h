#pragma once
#include <vector>

#include <mutex>
#include <thread>

#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

#include "Assets/Material.h"

struct Scene {
	Camera camera;

	int    mesh_count;
	Mesh * meshes;

	std::vector<const MeshData *> mesh_datas;

	std::vector<Material> materials;
	std::vector<Texture>  textures;

	std::mutex       textures_mutex;
	std::atomic<int> num_textures_finished;

	Sky sky;
	
	bool has_diffuse    = false;
	bool has_dielectric = false;
	bool has_glossy     = false;
	bool has_lights     = false;

	void init(int mesh_count, const char * mesh_names[], const char * sky_name);

	void wait_until_textures_loaded();

	void check_materials();

	void update(float delta);
};
