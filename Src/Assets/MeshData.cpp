#include "MeshData.h"

#include <GL/glew.h>

#include <unordered_map>

#include "Assets/OBJLoader.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/SBVHBuilder.h"
#include "BVH/Builders/QBVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"
#include "BVH/BVHOptimizer.h"

#include "Util/Util.h"
#include "Util/ScopeTimer.h"

static constexpr int UNDERLYING_BVH_TYPE    = BVH_TYPE == BVH_SBVH ? BVH_SBVH : BVH_BVH; // All BVH use standard BVH as underlying type, only SBVH uses SBVH
static constexpr int MAX_PRIMITIVES_IN_LEAF = BVH_TYPE == BVH_CWBVH || BVH_ENABLE_OPTIMIZATION ? 1 : INT_MAX; // CWBVH and BVH optimization require 1 primitive per leaf Node, the others have no upper limits

static constexpr const char * BVH_FILE_EXTENSION = ".bvh";
static constexpr int          BVH_FILETYPE_VERSION = 2;

struct BVHFileHeader {
	char filetype_identifier[4];
	char filetype_version;

	// Store settings with which the BVH was created
	char underlying_bvh_type;
	bool bvh_is_optimized;
	int  max_primitives_in_leaf;
	float sah_cost_node;
	float sah_cost_leaf;

	int num_triangles;
	int num_nodes;
	int num_indices;
};

struct Vertex {
	Vector3 position;
	Vector3 normal;
	Vector2 uv;
	int     triangle_id;
};

static std::unordered_map<std::string, int> cache;

static void save_to_disk(const BVH & bvh, const MeshData * mesh_data, const char * filename) {
	int    bvh_filename_length = strlen(filename) + strlen(BVH_FILE_EXTENSION) + 1;
	char * bvh_filename        = MALLOCA(char, bvh_filename_length);

	strcpy_s(bvh_filename, bvh_filename_length, filename);
	strcat_s(bvh_filename, bvh_filename_length, BVH_FILE_EXTENSION);

	FILE * file;
	fopen_s(&file, bvh_filename, "wb");

	if (file == nullptr) {
		printf("WARNING: Unable to save BVH to file %s!\n", bvh_filename);

		FREEA(bvh_filename);

		return;
	}

	BVHFileHeader header = { };
	header.filetype_identifier[0] = 'B';
	header.filetype_identifier[1] = 'V';
	header.filetype_identifier[2] = 'H';
	header.filetype_identifier[3] = '\0';
	header.filetype_version = BVH_FILETYPE_VERSION;

	header.underlying_bvh_type    = UNDERLYING_BVH_TYPE;
	header.bvh_is_optimized       = BVH_ENABLE_OPTIMIZATION;
	header.max_primitives_in_leaf = MAX_PRIMITIVES_IN_LEAF;
	header.sah_cost_node = SAH_COST_NODE;
	header.sah_cost_leaf = SAH_COST_LEAF;

	header.num_triangles = mesh_data->triangle_count;
	header.num_nodes     = bvh.node_count;
	header.num_indices   = bvh.index_count;

	fwrite(reinterpret_cast<const char *>(&header), sizeof(header), 1, file);

	fwrite(reinterpret_cast<const char *>(mesh_data->triangles), sizeof(Triangle), mesh_data->triangle_count, file);
	fwrite(reinterpret_cast<const char *>(bvh.nodes),            sizeof(BVHNode),  bvh.node_count,            file);
	fwrite(reinterpret_cast<const char *>(bvh.indices),          sizeof(int),      bvh.index_count,           file);

	fclose(file);

	FREEA(bvh_filename);
}

static bool try_to_load_from_disk(BVH & bvh, MeshData * mesh_data, const char * filename) {
	int    bvh_filename_size = strlen(filename) + strlen(BVH_FILE_EXTENSION) + 1;
	char * bvh_filename      = MALLOCA(char, bvh_filename_size);

	strcpy_s(bvh_filename, bvh_filename_size, filename);
	strcat_s(bvh_filename, bvh_filename_size, BVH_FILE_EXTENSION);

	// If the BVH file doesn't exist or is outdated return false
	if (!Util::file_exists(bvh_filename) || !Util::file_is_newer(filename, bvh_filename)) {
		FREEA(bvh_filename);

		return false;
	}

	FILE * file;
	fopen_s(&file, bvh_filename, "rb");
	
	BVHFileHeader header = { };

	bool success = false;

	if (!file) {
		printf("WARNING: Unable to open BVH file '%s'!\n", bvh_filename);
		goto exit;
	}

	fread(reinterpret_cast<char *>(&header), sizeof(header), 1, file);

	if (strcmp(header.filetype_identifier, "BVH") != 0) {
		printf("WARNING: BVH file '%s' has an invalid header!\n", bvh_filename);
		goto exit;
	}

	if (header.filetype_version < BVH_FILETYPE_VERSION) goto exit;

	// Check if the settings used to create the BVH file are the same as the current settings
	if (header.underlying_bvh_type    != UNDERLYING_BVH_TYPE || 
		header.bvh_is_optimized       != BVH_ENABLE_OPTIMIZATION || 
		header.max_primitives_in_leaf != MAX_PRIMITIVES_IN_LEAF ||
		header.sah_cost_node != SAH_COST_NODE || 
		header.sah_cost_leaf != SAH_COST_LEAF
	) {
		printf("BVH file '%s' was created with different settings, rebuiling BVH from scratch.\n", bvh_filename);
		goto exit;
	}

	mesh_data->triangle_count = header.num_triangles;
	bvh.node_count            = header.num_nodes;
	bvh.index_count           = header.num_indices;

	mesh_data->triangles = new Triangle[mesh_data->triangle_count];
	bvh.nodes            = new BVHNode[bvh.node_count];
	bvh.indices          = new int    [bvh.index_count];

	fread(reinterpret_cast<char *>(mesh_data->triangles), sizeof(Triangle), mesh_data->triangle_count, file);
	fread(reinterpret_cast<char *>(bvh.nodes),            sizeof(BVHNode),  bvh.node_count,            file);
	fread(reinterpret_cast<char *>(bvh.indices),          sizeof(int),      bvh.index_count,           file);

	printf("Loaded BVH %s from disk\n", bvh_filename);

	success = true;

exit:
	fclose(file);

	FREEA(bvh_filename);

	return success;
}

int MeshData::load(const char * filename) {
	int & mesh_data_index = cache[filename];

	// If the cache already contains this Model Data simply return it
	if (mesh_data_index != 0) return mesh_data_index - 1;

	mesh_data_index = mesh_datas.size() + 1;

	MeshData * mesh_data = new MeshData();
	mesh_datas.push_back(mesh_data);
	
	BVH bvh;
	bool bvh_loaded = try_to_load_from_disk(bvh, mesh_data, filename);

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
		
#if BVH_TYPE == BVH_SBVH // All other BVH types use standard BVH as a starting point
		{
			ScopeTimer timer("SBVH Construction");

			SBVHBuilder sbvh_builder;
			sbvh_builder.init(&bvh, mesh_data->triangle_count, MAX_PRIMITIVES_IN_LEAF);
			sbvh_builder.build(mesh_data->triangles, mesh_data->triangle_count);
			sbvh_builder.free();
		}
#else
		{
			ScopeTimer timer("BVH Construction");
			
			BVHBuilder bvh_builder;
			bvh_builder.init(&bvh, mesh_data->triangle_count, MAX_PRIMITIVES_IN_LEAF);
			bvh_builder.build(mesh_data->triangles, mesh_data->triangle_count);
			bvh_builder.free();
		}
#endif
		
#if BVH_ENABLE_OPTIMIZATION
		BVHOptimizer::optimize(bvh);
#endif

		save_to_disk(bvh, mesh_data, filename);
	}
	
#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	mesh_data->bvh = bvh;
#elif BVH_TYPE == BVH_QBVH
	// Collapse binary BVH into quaternary BVH
	QBVHBuilder qbvh_builder;
	qbvh_builder.init(&mesh_data->bvh, bvh);
	qbvh_builder.build(bvh);
	
	delete [] bvh.nodes;
#elif BVH_TYPE == BVH_CWBVH
	// Collapse binary BVH into 8-way Compressed Wide BVH
	CWBVHBuilder cwbvh_builder;
	cwbvh_builder.init(&mesh_data->bvh, bvh);
	cwbvh_builder.build(bvh);
	cwbvh_builder.free();

	delete [] bvh.indices;
	delete [] bvh.nodes;
#endif
	
	return mesh_data_index - 1;
}

void MeshData::gl_init(int reverse_indices[]) const {
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

	glGenVertexArrays(1, &gl_vao);
	glBindVertexArray(gl_vao);
	
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glEnableVertexAttribArray(2);
	glEnableVertexAttribArray(3);
	
	glVertexAttribFormat (0, 3, GL_FLOAT, false, offsetof(Vertex, position));
	glVertexAttribFormat (1, 3, GL_FLOAT, false, offsetof(Vertex, normal));
	glVertexAttribFormat (2, 2, GL_FLOAT, false, offsetof(Vertex, uv));
	glVertexAttribIFormat(3, 1, GL_INT,          offsetof(Vertex, triangle_id));

	glVertexAttribBinding(0, 0);
	glVertexAttribBinding(1, 0);
	glVertexAttribBinding(2, 0);
	glVertexAttribBinding(3, 0);

	glGenBuffers(1, &gl_vbo);
	glBindBuffer(GL_ARRAY_BUFFER, gl_vbo);
	glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(Vertex), vertices, GL_STATIC_DRAW);

	glBindVertexArray(0);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
	glDisableVertexAttribArray(2);
	glDisableVertexAttribArray(3);

	delete [] vertices;
}

void MeshData::gl_render() const {
	glBindVertexArray(gl_vao);
	glBindVertexBuffer(0, gl_vbo, 0, sizeof(Vertex));

	glDrawArrays(GL_TRIANGLES, 0, triangle_count * 3);
}
