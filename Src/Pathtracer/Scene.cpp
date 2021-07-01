#include "Scene.h"

#include <ctype.h>

#include "Assets/Material.h"
#include "Assets/OBJLoader.h"
#include "Assets/MitsubaLoader.h"

#include "Util/Util.h"

void Scene::init(const char * scene_name, const char * sky_name) {
	camera.init(DEG_TO_RAD(110.0f));

	// Default Material
	Material default_material = { };
	default_material.name    = "Default";
	default_material.diffuse = Vector3(1.0f, 0.0f, 1.0f);
	asset_manager.add_material(default_material);

	const char * file_extension = Util::file_get_extension(scene_name);

	if (strcmp(file_extension, "obj") == 0) {
		MeshDataHandle mesh_data_id = asset_manager.add_mesh_data(scene_name);

		Mesh & mesh = meshes.emplace_back();
		mesh.init(scene_name, mesh_data_id, *this);
		mesh.material_id = MaterialHandle::get_default();
	} else if (strcmp(file_extension, "xml") == 0) {
		MitsubaLoader::load(scene_name, *this);		
	} else abort();

	// Initialize Sky
	sky.init(sky_name);
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
