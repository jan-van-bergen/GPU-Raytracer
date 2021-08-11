#include "OBJLoader.h"

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

#include "Assets/Texture.h"
#include "Assets/Material.h"

#include "Pathtracer/Scene.h"

#include "Util/Util.h"
#include "Util/ScopeTimer.h"

bool OBJLoader::load(const char * filename, Triangle *& triangles, int & triangle_count) {
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warning;
	std::string error;

	char path[512];	Util::get_path(filename, path);

	bool success = tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, filename, path);
	if (!success) {
		printf("ERROR: Unable to open obj file %s!\n", filename);
		abort();
		return false;
	}

	// Load Geometry
	int total_vertex_count = 0;
	
	// Count total amount of vertices over all Shapes
	for (int s = 0; s < shapes.size(); s++) {
		int vertex_count = shapes[s].mesh.indices.size();
		total_vertex_count += vertex_count;
	}

	triangle_count = total_vertex_count / 3;
	triangles      = new Triangle[triangle_count];

	int triangle_offset = 0;

	for (int s = 0; s < shapes.size(); s++) {
		int vertex_count = shapes[s].mesh.indices.size();
		assert(vertex_count % 3 == 0);

		// Iterate over faces
		for (int f = 0; f < vertex_count / 3; f++) {
			Vector3 positions [3] = { };
			Vector2 tex_coords[3] = { };
			Vector3 normals   [3] = { };

			for (int i = 0; i < 3; i++) {
				int vertex_index    = shapes[s].mesh.indices[3*f + i].vertex_index;
				int tex_coord_index = shapes[s].mesh.indices[3*f + i].texcoord_index;
				int normal_index    = shapes[s].mesh.indices[3*f + i].normal_index;
			
				if (vertex_index != INVALID) {
					positions[i] = Vector3(
						attrib.vertices[3*vertex_index    ], 
						attrib.vertices[3*vertex_index + 1], 
						attrib.vertices[3*vertex_index + 2]
					);
				}
				if (tex_coord_index != INVALID) {
					tex_coords[i] = Vector2(
							   attrib.texcoords[2*tex_coord_index    ], 
						1.0f - attrib.texcoords[2*tex_coord_index + 1] // Flip uv along y
					);
				}
				if (normal_index != INVALID) {
					normals[i] = Vector3(
						attrib.normals[3*normal_index    ], 
						attrib.normals[3*normal_index + 1], 
						attrib.normals[3*normal_index + 2]
					);
				}
			}

			triangles[triangle_offset + f].position_0 = positions[0];
			triangles[triangle_offset + f].position_1 = positions[1];
			triangles[triangle_offset + f].position_2 = positions[2];

			bool normal_0_invalid = Math::approx_equal(Vector3::length(normals[0]), 0.0f);
			bool normal_1_invalid = Math::approx_equal(Vector3::length(normals[1]), 0.0f);
			bool normal_2_invalid = Math::approx_equal(Vector3::length(normals[2]), 0.0f);

			// Replace zero normals with the geometric normal of defined by the Triangle
			if (normal_0_invalid || normal_1_invalid || normal_2_invalid) {
				Vector3 geometric_normal = Vector3::normalize(Vector3::cross(
					triangles[triangle_offset + f].position_1 - triangles[triangle_offset + f].position_0,
					triangles[triangle_offset + f].position_2 - triangles[triangle_offset + f].position_0
				));

				if (normal_0_invalid) normals[0] = geometric_normal;
				if (normal_1_invalid) normals[1] = geometric_normal;
				if (normal_2_invalid) normals[2] = geometric_normal;
			} 

			triangles[triangle_offset + f].normal_0 = normals[0];
			triangles[triangle_offset + f].normal_1 = normals[1];
			triangles[triangle_offset + f].normal_2 = normals[2];

			triangles[triangle_offset + f].tex_coord_0 = tex_coords[0];
			triangles[triangle_offset + f].tex_coord_1 = tex_coords[1];
			triangles[triangle_offset + f].tex_coord_2 = tex_coords[2];

			triangles[triangle_offset + f].calc_aabb();
		}

		triangle_offset += vertex_count / 3;
	}

	assert(triangle_offset == triangle_count);

	printf("Loaded OBJ %s from disk (%i triangles)\n", filename, triangle_count);

	return true;
}
