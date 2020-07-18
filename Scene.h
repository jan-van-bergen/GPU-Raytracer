#pragma once
#include "Camera.h"
#include "Mesh.h"
#include "Sky.h"

struct Scene {
	Camera camera;

	int    mesh_count;
	Mesh * meshes;

	Sky sky;
	
	bool has_diffuse;
	bool has_dielectric;
	bool has_glossy;
	bool has_lights;

	void init(const char * mesh_names[], int mesh_count, const char * sky_name);

	void update(float delta);
};
