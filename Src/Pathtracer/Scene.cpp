#include "Scene.h"

#include <ctype.h>
#include <stdio.h>

#include "Assets/Material.h"
#include "Assets/OBJLoader.h"
#include "Assets/PLYLoader.h"
#include "Assets/MitsubaLoader.h"

#include "Util/Util.h"

void Scene::init(const SceneConfig & scene_config) {
	camera.init(DEG_TO_RAD(110.0f));

	asset_manager.init();

	// Default Material
	Material default_material = { };
	default_material.name    = "Default";
	default_material.diffuse = Vector3(1.0f, 0.0f, 1.0f);
	asset_manager.add_material(default_material);

	for (int i = 0; i < scene_config.scenes.size(); i++) {
		const char * scene_name = scene_config.scenes[i];

		const char * file_extension = Util::find_last(scene_name, ".");
		if (!file_extension) {
			printf("ERROR: File '%s' has no file extension, cannot deduce file format!\n", scene_name);
			abort();
		}

		if (strcmp(file_extension, "obj") == 0) {
			add_mesh(scene_name, asset_manager.add_mesh_data(scene_name, OBJLoader::load));
		} else if (strcmp(file_extension, "ply") == 0) {
			add_mesh(scene_name, asset_manager.add_mesh_data(scene_name, PLYLoader::load));
		} else if (strcmp(file_extension, "xml") == 0) {
			MitsubaLoader::load(scene_name, *this);
		} else {
			printf("ERROR: '%s' file format is not supported!\n", file_extension);
			abort();
		}
	}

	// Initialize Sky
	sky.init(scene_config.sky);
}

Mesh & Scene::add_mesh(const char * name, MeshDataHandle mesh_data_handle, MaterialHandle material_handle) {
	Mesh & mesh = meshes.emplace_back();
	mesh.init(name, mesh_data_handle, material_handle, *this);

	return mesh;
}

void Scene::check_materials() {
	has_diffuse    = false;
	has_dielectric = false;
	has_glossy     = false;
	has_lights     = false;

	// Check properties of the Scene, so we know which kernels are required
	for (int i = 0; i < asset_manager.materials.size(); i++) {
		switch (asset_manager.materials[i].type) {
			case Material::Type::DIFFUSE:    has_diffuse    = true; break;
			case Material::Type::DIELECTRIC: has_dielectric = true; break;
			case Material::Type::GLOSSY:     has_glossy     = true; break;
			case Material::Type::LIGHT:      has_lights     = true; break;
		}
	}
}

void Scene::update(float delta) {
	for (int i = 0; i < meshes.size(); i++) {
		meshes[i].update();
	}
}
