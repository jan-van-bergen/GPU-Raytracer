#include "Scene.h"

#include <ctype.h>

#include "Assets/Material.h"
#include "Assets/OBJLoader.h"
#include "Assets/MitsubaLoader.h"

#include "Util/Util.h"

void Scene::init(const char * scene_name, const char * sky_name) {
	camera.init(DEG_TO_RAD(110.0f));

	// Default Material
	Material & default_material = materials.emplace_back();
	default_material.name    = "Default";
	default_material.diffuse = Vector3(1.0f, 0.0f, 1.0f);

	const char * file_extension = Util::file_get_extension(scene_name);

	if (strcmp(file_extension, "obj") == 0) {
		Mesh & mesh = meshes.emplace_back();
		mesh.init(scene_name, MeshData::load(scene_name, *this), *this);
		materials.emplace_back();
	} else if (strcmp(file_extension, "xml") == 0) {
		MitsubaLoader::load(scene_name, *this);		
	} else abort();

	// Initialize Sky
	sky.init(sky_name);
}

void Scene::wait_until_textures_loaded() {
	using namespace std::chrono_literals;

	while (num_textures_finished < textures.size()) {
		std::this_thread::sleep_for(100ms);
	}
}

void Scene::check_materials() {
	has_diffuse    = false;
	has_dielectric = false;
	has_glossy     = false;
	has_lights     = false;

	// Check properties of the Scene, so we know which kernels are required
	for (int i = 0; i < materials.size(); i++) {
		switch (materials[i].type) {
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
