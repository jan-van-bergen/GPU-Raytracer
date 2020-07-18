#include "MeshData.h"

#include <GL/glew.h>

#include <unordered_map>

#include "OBJLoader.h"
#include "BVHBuilders.h"

#include "Util.h"
#include "ScopeTimer.h"

struct Vertex {
	Vector3 position;
	Vector3 normal;
	Vector2 uv;
	int     triangle_id;
};

static std::unordered_map<std::string, int> cache;

static void save_to_disk(const BVH & bvh, const MeshData * mesh_data, const char * filename, const char * file_extension) {
	assert(file_extension[0] == '.');

	int    bvh_filename_length = strlen(filename) + strlen(file_extension) + 1;
	char * bvh_filename        = MALLOCA(char, bvh_filename_length);

	strcpy_s(bvh_filename, bvh_filename_length, filename);
	strcat_s(bvh_filename, bvh_filename_length, file_extension);

	FILE * file;
	fopen_s(&file, bvh_filename, "wb");

	if (file == nullptr) {
		printf("WARNING: Unable to save BVH to file %s!\n", bvh_filename);

		FREEA(bvh_filename);

		return;
	}

	fwrite(reinterpret_cast<const char *>(&mesh_data->triangle_count), sizeof(int),      1,                         file);
	fwrite(reinterpret_cast<const char *>( mesh_data->triangles),      sizeof(Triangle), mesh_data->triangle_count, file);

	fwrite(reinterpret_cast<const char *>(&bvh.node_count), sizeof(int),     1,              file);
	fwrite(reinterpret_cast<const char *>( bvh.nodes),      sizeof(BVHNode), bvh.node_count, file);

	fwrite(reinterpret_cast<const char *>(&bvh.index_count), sizeof(int), 1,               file);
	fwrite(reinterpret_cast<const char *>( bvh.indices),     sizeof(int), bvh.index_count, file);

	fclose(file);

	FREEA(bvh_filename);
}

static bool try_to_load_from_disk(BVH & bvh, MeshData * mesh_data, const char * filename, const char * file_extension) {
	assert(file_extension[0] == '.');

	int    bvh_filename_size = strlen(filename) + strlen(file_extension) + 1;
	char * bvh_filename      = MALLOCA(char, bvh_filename_size);

	strcpy_s(bvh_filename, bvh_filename_size, filename);
	strcat_s(bvh_filename, bvh_filename_size, file_extension);

	if (!Util::file_exists(bvh_filename) || !Util::file_is_newer(filename, bvh_filename)) {
		FREEA(bvh_filename);

		return false;
	}

	FILE * file;
	fopen_s(&file, bvh_filename, "rb");

	if (!file) {
		FREEA(bvh_filename);

		return false;
	}

	fread(reinterpret_cast<char *>(&mesh_data->triangle_count), sizeof(int), 1, file);

	mesh_data->triangles = new Triangle[mesh_data->triangle_count];
	fread(reinterpret_cast<char *>(mesh_data->triangles), sizeof(Triangle), mesh_data->triangle_count, file);
		
	fread(reinterpret_cast<char *>(&bvh.node_count), sizeof(int), 1, file);

	bvh.nodes = new BVHNode[bvh.node_count];
	fread(reinterpret_cast<char *>(bvh.nodes), sizeof(BVHNode), bvh.node_count, file);

	fread(reinterpret_cast<char *>(&bvh.index_count), sizeof(int), 1, file);
			
	bvh.indices = new int[bvh.index_count];
	fread(reinterpret_cast<char *>(bvh.indices), sizeof(int), bvh.index_count, file);

	fclose(file);

	printf("Loaded BVH  %s from disk\n", bvh_filename);

	FREEA(bvh_filename);

	return true;
}

int MeshData::load(const char * filename) {
	int & mesh_data_index = cache[filename];

	// If the cache already contains this Model Data simply return it
	if (mesh_data_index != 0 && mesh_datas.size() > 0 ) return mesh_data_index;

#if BVH_TYPE == BVH_BVH
	const char * file_extension = ".bvh";
#else
	const char * file_extension = ".sbvh";
#endif

	mesh_data_index = mesh_datas.size();

	MeshData * mesh_data = new MeshData();
	mesh_datas.push_back(mesh_data);

	BVH bvh;
	bool bvh_loaded = try_to_load_from_disk(bvh, mesh_data, filename, file_extension);

	if (bvh_loaded) {
		// If the BVH loaded successfully we only need to load the Materials
		// of the Mesh, because the geometry is already included in the BVH

		// Replace ".obj" in the filename with ".mtl"
		int filename_length = strlen(filename);
		char * mtl_filename = MALLOCA(char, filename_length + 1);
		strcpy_s(mtl_filename, filename_length + 1, filename);
		memcpy(mtl_filename + filename_length - 4, ".mtl", 4);

		OBJLoader::load_mtl(mtl_filename, mesh_data);

		FREEA(mtl_filename);
	} else {
		OBJLoader::load_obj(filename, mesh_data);

#if BVH_TYPE == BVH_BVH
		{
			ScopeTimer timer("BVH Construction");

			bvh = BVHBuilders::build_bvh(mesh_data->triangles, mesh_data->triangle_count);
		}
#else
		{
			ScopeTimer timer("SBVH Construction");

			bvh = BVHBuilders::build_sbvh(mesh_data->triangles, mesh_data->triangle_count);
		}
#endif

		save_to_disk(bvh, mesh_data, filename, file_extension);
	}

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	mesh_data->bvh = bvh;
#elif BVH_TYPE == BVH_QBVH
	mesh_data->bvh = BVHBuilders::qbvh_from_binary_bvh(bvh);
#elif BVH_TYPE == BVH_CWBVH
	mesh_data->bvh = BVHBuilders::cwbvh_from_binary_bvh(bvh);
#endif

	return mesh_data_index;
}

void MeshData::init_gl(int reverse_indices[]) const {
	int      vertex_count = triangle_count * 3;
	Vertex * vertices = new Vertex[vertex_count];

	for (int t = 0; t < triangle_count; t++) {
		int index_0 = 3 * t;
		int index_1 = 3 * t + 1;
		int index_2 = 3 * t + 2;

		vertices[index_0].position = triangles[t].position_0;
		vertices[index_1].position = triangles[t].position_1;
		vertices[index_2].position = triangles[t].position_2;

		vertices[index_0].normal = triangles[t].normal_0;
		vertices[index_1].normal = triangles[t].normal_1;
		vertices[index_2].normal = triangles[t].normal_2;

		// Barycentric coordinates
		vertices[index_0].uv = Vector2(0.0f, 0.0f);
		vertices[index_1].uv = Vector2(1.0f, 0.0f);
		vertices[index_2].uv = Vector2(0.0f, 1.0f);

		vertices[index_0].triangle_id = reverse_indices[t];
		vertices[index_1].triangle_id = reverse_indices[t];
		vertices[index_2].triangle_id = reverse_indices[t];
	}

	glGenBuffers(1, &gl_vbo);

	glBindBuffer(GL_ARRAY_BUFFER, gl_vbo);
	glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(Vertex), vertices, GL_STATIC_DRAW);

	delete [] vertices;
}

void MeshData::render() const {
	glBindBuffer(GL_ARRAY_BUFFER, gl_vbo);

	glVertexAttribPointer (0, 3, GL_FLOAT, false, sizeof(Vertex), reinterpret_cast<const GLvoid *>(offsetof(Vertex, position)));
	glVertexAttribPointer (1, 3, GL_FLOAT, false, sizeof(Vertex), reinterpret_cast<const GLvoid *>(offsetof(Vertex, normal)));
	glVertexAttribPointer (2, 2, GL_FLOAT, false, sizeof(Vertex), reinterpret_cast<const GLvoid *>(offsetof(Vertex, uv)));
	glVertexAttribIPointer(3, 1, GL_INT,          sizeof(Vertex), reinterpret_cast<const GLvoid *>(offsetof(Vertex, triangle_id)));

	glDrawArrays(GL_TRIANGLES, 0, triangle_count * 3);
}
