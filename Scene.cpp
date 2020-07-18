#include "Scene.h"

#include <ctype.h>

#include "Material.h"

#include "Util.h"

void Scene::init(const char * mesh_names[], int mesh_count, const char * sky_name) {
	if (mesh_count == 0) {
		puts("ERROR: No Meshes provided!");
		abort();
	}

	camera.init(DEG_TO_RAD(110.0f));
	camera.resize(SCREEN_WIDTH, SCREEN_HEIGHT);

	// Set default Material before loading Meshes
	Material & default_material = Material::materials.emplace_back();
	default_material.diffuse = Vector3(1.0f, 0.0f, 1.0f);
	
	// Load Meshes
	this->mesh_count = mesh_count;
	this->meshes     = new Mesh[mesh_count];
	
	for (int i = 0; i < mesh_count; i++) {
		meshes[i].init(MeshData::load(mesh_names[i]));
	}
	
	has_diffuse    = false;
	has_dielectric = false;
	has_glossy     = false;
	has_lights     = false;

	// Check properties of the Scene, so we know which kernels are required
	for (int i = 0; i < Material::materials.size(); i++) {
		switch (Material::materials[i].type) {
			case Material::Type::DIFFUSE:    has_diffuse    = true; break;
			case Material::Type::DIELECTRIC: has_dielectric = true; break;
			case Material::Type::GLOSSY:     has_glossy     = true; break;
			case Material::Type::LIGHT:      has_lights     = true; break;
		}
	}

	printf("\nScene info:\ndiffuse:    %s\ndielectric: %s\nglossy:     %s\nlights:     %s\n\n", 
		has_diffuse    ? "yes" : "no",
		has_dielectric ? "yes" : "no",
		has_glossy     ? "yes" : "no",
		has_lights     ? "yes" : "no"
	);

	// Initialize Sky
	sky.init(sky_name);
	
	// Initialize Camera position/orientation based on the Scene name
	int    scene_name_length = strlen(mesh_names[0]);
	char * scene_name_lower  = MALLOCA(char, scene_name_length + 1);

	for (int i = 0; i < scene_name_length + 1; i++) {
		scene_name_lower[i] = tolower(mesh_names[0][i]);
	}

	if (strstr(scene_name_lower, "pica.obj")) {
		camera.position = Vector3(-7.640668f, 16.404673f, 17.845022f);
		camera.rotation = Quaternion(-0.256006f, -0.069205f, -0.018378f, 0.964019f);	
	} else if (strstr(scene_name_lower, "sponza.obj")) {
		camera.position = Vector3(116.927467f, 15.586369f, -2.997146f);
		camera.rotation = Quaternion(0.000000f, 0.692966f, 0.000000f, 0.720969f);
	} else if (strstr(scene_name_lower, "scene.obj")) {
		camera.position = Vector3(-0.126737f, 0.613379f, 3.716630f);
		camera.rotation = Quaternion(-0.107255f, -0.002421f, 0.000262f, -0.994227f);
	} else if (strstr(scene_name_lower, "cornellbox.obj")) {
		camera.position = Vector3(0.528027f, 1.004323f, -0.774033f);
		camera.rotation = Quaternion(0.035059f, -0.963870f, 0.208413f, 0.162142f);
	} else if (strstr(scene_name_lower, "glossy.obj")) {
		camera.position = Vector3(-5.438800f, 5.910520f, -7.185338f);
		camera.rotation = Quaternion(0.242396f, 0.716713f, 0.298666f, -0.581683f);
	} else if (strstr(scene_name_lower, "bunny.obj")) {
		camera.position = Vector3(-27.662603f, 26.719784f, -15.835464f);
		camera.rotation = Quaternion(0.076750f, 0.900785f, 0.177892f, -0.388638f);
	} else if (strstr(scene_name_lower, "test.obj")) {
		camera.position = Vector3(4.157419f, 4.996608f, 8.337481f);
		camera.rotation = Quaternion(0.000000f, 0.310172f, 0.000000f, 0.950679f);
	} else if (strstr(scene_name_lower, "bistro.obj")) {
		camera.position = Vector3(-13.665823f, 2.480730f, -2.920546f);
		camera.rotation = Quaternion(0.000000f, -0.772662f, 0.000000f, 0.634818f);
	} else if (strstr(scene_name_lower, "rungholt.obj")) {
		camera.position = Vector3(-22.413084f, 18.681219f, -23.566196f);
		camera.rotation = Quaternion(0.000000f, 0.716948f, 0.000000f, -0.697125f);
	} else {
		camera.position = Vector3(1.272743f, 3.097532f, -3.189943f);
		camera.rotation = Quaternion(0.000000f, 0.995683f, 0.000000f, -0.092814f);
	}

	//camera.position = Vector3(23.244606f, 1.007305f, 0.978476f);
	//camera.rotation = Quaternion(-0.109851f, -0.454757f, -0.056640f, 0.881994f);

	//camera.position = Vector3(24.923054f, 1.559265f, -0.216484f);
	//camera.rotation = Quaternion(-0.498747f, -0.500471f, -0.480318f, 0.519675f);

	FREEA(scene_name_lower);
}

void Scene::update(float delta) {
	meshes[1].position.x = 100.0f;

	//meshes[2].position.y  = 1.5f;
	//meshes[2].position.x += 0.5f * delta;
	//
	//meshes[3].position.y += delta;

	static float angle = 0.0f;
	angle += 0.5f * delta;
	meshes[1].rotation = Quaternion::axis_angle(Vector3(0.0f, 1.0f, 0.0f), angle);

	for (int i = 0; i < mesh_count; i++) {
		meshes[i].update();
	}
}
