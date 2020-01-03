#include "MeshData.h"

#include <vector>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader/tiny_obj_loader.h>

#include "Util.h"
#include "ScopedTimer.h"

#define MESH_USE_BVH  0
#define MESH_USE_SBVH 1

#define MESH_ACCELERATOR MESH_USE_SBVH

static std::unordered_map<std::string, MeshData *> cache;

const MeshData * MeshData::load(const char * file_path) {
	MeshData *& mesh_data = cache[file_path];

	// If the cache already contains this Model Data simply return it
	if (mesh_data) return mesh_data;

	// Otherwise, load new MeshData
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;

	std::string warning;
	std::string error;

	const char * path = Util::get_path(file_path);

	tinyobj::LoadObj(&attrib, &shapes, &materials, &warning, &error, file_path, path);

	if (shapes.size() == 0) abort(); // Either the model is empty, or something went wrong
	
	mesh_data = new MeshData();
	
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
			mesh_data->triangles[triangle_offset + v].position0 = positions[3*v    ];
			mesh_data->triangles[triangle_offset + v].position1 = positions[3*v + 1];
			mesh_data->triangles[triangle_offset + v].position2 = positions[3*v + 2];

			mesh_data->triangles[triangle_offset + v].normal0 = normals[3*v    ];
			mesh_data->triangles[triangle_offset + v].normal1 = normals[3*v + 1];
			mesh_data->triangles[triangle_offset + v].normal2 = normals[3*v + 2];
			
			mesh_data->triangles[triangle_offset + v].tex_coord0 = tex_coords[3*v    ];
			mesh_data->triangles[triangle_offset + v].tex_coord1 = tex_coords[3*v + 1];
			mesh_data->triangles[triangle_offset + v].tex_coord2 = tex_coords[3*v + 2];

			int material_id = shapes[s].mesh.material_ids[v];
			if (material_id == INVALID) material_id = 0;
			
			//assert(material_id < material_count);

			//mesh_data->triangles[triangle_offset + v].material = &mesh_data->materials[material_id];
		}
		
		triangle_offset += vertex_count / 3;
	}

	assert(triangle_offset == mesh_data->triangle_count);

	printf("Loaded Mesh %s from disk, consisting of %u triangles.\n", file_path, mesh_data->triangle_count);
	
	delete [] positions;
	delete [] tex_coords;
	delete [] normals;

	delete [] path;

	return mesh_data;
}
