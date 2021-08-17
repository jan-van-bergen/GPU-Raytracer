#include "BVHLoader.h"

#include "Util/Util.h"

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

bool BVHLoader::try_to_load(const char * filename, MeshData & mesh_data, BVH & bvh) {
	int    bvh_filename_size = strlen(filename) + strlen(BVH_FILE_EXTENSION) + 1;
	char * bvh_filename      = MALLOCA(char, bvh_filename_size);

	if (!bvh_filename) return false;

	strcpy_s(bvh_filename, bvh_filename_size, filename);
	strcat_s(bvh_filename, bvh_filename_size, BVH_FILE_EXTENSION);

	// If the BVH file doesn't exist or is outdated return false
	if (!Util::file_exists(bvh_filename) || !Util::file_is_newer(filename, bvh_filename)) {
		FREEA(bvh_filename);

		return false;
	}

	FILE * file; fopen_s(&file, bvh_filename, "rb");

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

	if (header.filetype_version != BVH_FILETYPE_VERSION) goto exit;

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

	mesh_data.triangle_count = header.num_triangles;
	bvh.node_count           = header.num_nodes;
	bvh.index_count          = header.num_indices;

	mesh_data.triangles = new Triangle[mesh_data.triangle_count];
	bvh.nodes           = new BVHNode[bvh.node_count];
	bvh.indices         = new int    [bvh.index_count];

	fread(reinterpret_cast<char *>(mesh_data.triangles), sizeof(Triangle), mesh_data.triangle_count, file);
	fread(reinterpret_cast<char *>(bvh.nodes),           sizeof(BVHNode),  bvh.node_count,           file);
	fread(reinterpret_cast<char *>(bvh.indices),         sizeof(int),      bvh.index_count,          file);

	printf("Loaded BVH %s from disk\n", bvh_filename);

	success = true;

exit:
	if (file) fclose(file);

	FREEA(bvh_filename);
	return success;
}

bool BVHLoader::save(const char * filename, MeshData & mesh_data, BVH & bvh) {
	int    bvh_filename_length = strlen(filename) + strlen(BVH_FILE_EXTENSION) + 1;
	char * bvh_filename        = MALLOCA(char, bvh_filename_length);

	if (!bvh_filename) return false;

	strcpy_s(bvh_filename, bvh_filename_length, filename);
	strcat_s(bvh_filename, bvh_filename_length, BVH_FILE_EXTENSION);

	FILE * file; fopen_s(&file, bvh_filename, "wb");

	if (!file) {
		printf("WARNING: Unable to save BVH to file %s!\n", bvh_filename);

		FREEA(bvh_filename);
		return false;
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

	header.num_triangles = mesh_data.triangle_count;
	header.num_nodes     = bvh.node_count;
	header.num_indices   = bvh.index_count;

	fwrite(reinterpret_cast<const char *>(&header), sizeof(header), 1, file);

	fwrite(reinterpret_cast<const char *>(mesh_data.triangles), sizeof(Triangle), mesh_data.triangle_count, file);
	fwrite(reinterpret_cast<const char *>(bvh.nodes),           sizeof(BVHNode),  bvh.node_count,           file);
	fwrite(reinterpret_cast<const char *>(bvh.indices),         sizeof(int),      bvh.index_count,          file);

	fclose(file);

	FREEA(bvh_filename);
	return true;
}
