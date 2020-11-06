#include "OBJLoader.h"

#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

#include "Texture.h"
#include "Material.h"

#include "Util.h"
#include "ScopeTimer.h"

// Converts 'tinyobj::material_t' to 'Material'
// Returns offset into Material table to convert relative Material indices to global ones
static void load_materials(const std::vector<tinyobj::material_t> & materials, MeshData * mesh_data, const char * path) {
	mesh_data->material_offset = Material::materials.size();

	for (int i = 0; i < materials.size(); i++) {
		const tinyobj::material_t & material = materials[i];

		Material & new_material = Material::materials.emplace_back();

		switch (material.illum) {
			case 0: case 3:                 new_material.type = Material::Type::GLOSSY;     break;
			case 1: case 2:                 new_material.type = Material::Type::DIFFUSE;    break;
			case 4: case 5: case 6: case 7: new_material.type = Material::Type::DIELECTRIC; break;

			default: new_material.type = Material::Type::DIFFUSE;
		};

		new_material.diffuse = Vector3(material.diffuse[0], material.diffuse[1], material.diffuse[2]);
		if (material.diffuse_texname.length() > 0) {
			if (Util::file_exists(material.diffuse_texname.c_str())) {
				// Load as absolute path
				new_material.texture_id = Texture::load(material.diffuse_texname.c_str());
			} else {
				// Load as relative path
				new_material.texture_id = Texture::load((std::string(path) + material.diffuse_texname).c_str());
			}
		}

		new_material.emission = Vector3(material.emission);
		if (Vector3::length_squared(new_material.emission) > 0.0f) {
			new_material.type = Material::Type::LIGHT;
		}

		new_material.index_of_refraction = material.ior;
		new_material.absorption = Vector3(material.transmittance[0] - 1.0f, material.transmittance[1] - 1.0f, material.transmittance[2] - 1.0f);

		new_material.roughness = material.roughness * material.roughness;
	}
}

void OBJLoader::load_mtl(const char * filename, MeshData * mesh_data) {
	// Load only the mtl file
	std::map<std::string, int> material_map;
	std::vector<tinyobj::material_t> materials;

	std::string warning;
	std::string error;

	std::string str(filename);

	std::filebuf fb;
	if (fb.open(filename, std::ios::in)) {
		std::istream is(&fb);

		tinyobj::LoadMtl(&material_map, &materials, &is, &warning, &error);

		char * path = MALLOCA(char, strlen(filename) + 1);
		Util::get_path(filename, path);

		load_materials(materials, mesh_data, path);

		FREEA(path);
	} else {
		printf("ERROR: Cannot open mtl file %s! Make sure the .mtl file has the same name as the .obj file.\n", filename);
		abort();
	}
}

void OBJLoader::load_obj(const char * filename, MeshData * mesh_data) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warning;
	std::string error;

	char * path = MALLOCA(char, strlen(filename) + 1);
	Util::get_path(filename, path);

	bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, filename, path);
	if (!success) {
		printf("ERROR: Unable to open obj file %s!\n", filename);
		abort();
	}

	load_materials(materials, mesh_data, path);

	FREEA(path);
	
	// Load Meshes
	int total_vertex_count = 0;
	int max_vertex_count = -1;

	// Count total amount of vertices over all Shapes
	for (int s = 0; s < shapes.size(); s++) {
		int vertex_count = shapes[s].mesh.indices.size();
		total_vertex_count += vertex_count;

		if (vertex_count > max_vertex_count) {
			max_vertex_count = vertex_count;
		}
	}

	mesh_data->triangle_count = total_vertex_count / 3;
	mesh_data->triangles      = new Triangle[mesh_data->triangle_count];

	Vector3 * positions  = new Vector3[max_vertex_count];
	Vector2 * tex_coords = new Vector2[max_vertex_count];
	Vector3 * normals    = new Vector3[max_vertex_count];

	int triangle_offset = 0;

	for (int s = 0; s < shapes.size(); s++) {
		int vertex_count = shapes[s].mesh.indices.size();
		assert(vertex_count % 3 == 0);

		// Iterate over vertices and assign attributes
		for (int v = 0; v < vertex_count; v++) {
			int vertex_index    = shapes[s].mesh.indices[v].vertex_index;
			int tex_coord_index = shapes[s].mesh.indices[v].texcoord_index;
			int normal_index    = shapes[s].mesh.indices[v].normal_index;
			
			if (vertex_index != INVALID) {
				positions[v] = Vector3(
					attrib.vertices[3*vertex_index    ], 
					attrib.vertices[3*vertex_index + 1], 
					attrib.vertices[3*vertex_index + 2]
				);
			}
			if (tex_coord_index != INVALID) {
				tex_coords[v] = Vector2(
					       attrib.texcoords[2*tex_coord_index    ], 
					1.0f - attrib.texcoords[2*tex_coord_index + 1] // Flip uv along y
				);
			}
			if (normal_index != INVALID) {
				normals[v] = Vector3(
					attrib.normals[3*normal_index    ], 
					attrib.normals[3*normal_index + 1], 
					attrib.normals[3*normal_index + 2]
				);
			}
		}

		// Iterate over faces
		for (int v = 0; v < vertex_count / 3; v++) {
			mesh_data->triangles[triangle_offset + v].position_0 = positions[3*v    ];
			mesh_data->triangles[triangle_offset + v].position_1 = positions[3*v + 1];
			mesh_data->triangles[triangle_offset + v].position_2 = positions[3*v + 2];

			Vector3 normal_0 = normals[3*v    ];
			Vector3 normal_1 = normals[3*v + 1];
			Vector3 normal_2 = normals[3*v + 2];
			
			bool normal_0_invalid = Vector3::length(normal_0) == 0;
			bool normal_1_invalid = Vector3::length(normal_1) == 0;
			bool normal_2_invalid = Vector3::length(normal_2) == 0;

			// Replace zero normals with the geometric normal of defined by the Triangle
			if (normal_0_invalid || normal_1_invalid || normal_2_invalid) {
				Vector3 geometric_normal = Vector3::normalize(Vector3::cross(
					mesh_data->triangles[triangle_offset + v].position_1 - mesh_data->triangles[triangle_offset + v].position_0,
					mesh_data->triangles[triangle_offset + v].position_2 - mesh_data->triangles[triangle_offset + v].position_0
				));

				if (normal_0_invalid) normal_0 = geometric_normal;
				if (normal_1_invalid) normal_1 = geometric_normal;
				if (normal_2_invalid) normal_2 = geometric_normal;
			} 

			mesh_data->triangles[triangle_offset + v].normal_0 = normal_0;
			mesh_data->triangles[triangle_offset + v].normal_1 = normal_1;
			mesh_data->triangles[triangle_offset + v].normal_2 = normal_2;

			mesh_data->triangles[triangle_offset + v].tex_coord_0 = tex_coords[3*v    ];
			mesh_data->triangles[triangle_offset + v].tex_coord_1 = tex_coords[3*v + 1];
			mesh_data->triangles[triangle_offset + v].tex_coord_2 = tex_coords[3*v + 2];

			int material_id = shapes[s].mesh.material_ids[v];
			if (material_id == INVALID) material_id = 0;

			mesh_data->triangles[triangle_offset + v].material_id = material_id;
		}
		
		triangle_offset += vertex_count / 3;
	}

	assert(triangle_offset == mesh_data->triangle_count);

	// Calculate AABB for every Triangle
	for (int i = 0; i < mesh_data->triangle_count; i++) {
		Vector3 vertices[3] = { 
			mesh_data->triangles[i].position_0, 
			mesh_data->triangles[i].position_1, 
			mesh_data->triangles[i].position_2
		};
		mesh_data->triangles[i].aabb = AABB::from_points(vertices, 3);
	}

	printf("Loaded Mesh %s from disk, consisting of %u triangles.\n", filename, mesh_data->triangle_count);
	
	delete [] positions;
	delete [] tex_coords;
	delete [] normals;
}
