#include "BVHLoader.h"

#include "Util/Util.h"

const char * BVHLoader::get_bvh_filename(const char * filename) {
	int    bvh_filename_size = strlen(filename) + strlen(BVH_FILE_EXTENSION) + 1;
	char * bvh_filename      = new char[bvh_filename_size];

	strcpy_s(bvh_filename, bvh_filename_size, filename);
	strcat_s(bvh_filename, bvh_filename_size, BVH_FILE_EXTENSION);

	return bvh_filename;
}

struct BVHFileHeader {
	char filetype_identifier[4];
	char filetype_version;

	// Store settings with which the BVH was created
	char underlying_bvh_type;
	bool bvh_is_optimized;
	float sah_cost_node;
	float sah_cost_leaf;

	int num_triangles;
	int num_nodes;
	int num_indices;
};

bool BVHLoader::try_to_load(const char * filename, const char * bvh_filename, MeshData & mesh_data, BVH & bvh) {
	if (config.bvh_force_rebuild || !Util::file_exists(bvh_filename) || !Util::file_is_newer(filename, bvh_filename)) {
		return false;
	}

	FILE * file; fopen_s(&file, bvh_filename, "rb");

	BVHFileHeader header = { };
	bool success = false;

	if (!file) {
		printf("WARNING: Unable to open BVH file '%s'!\n", bvh_filename);
		return false;
	}

	fread(reinterpret_cast<char *>(&header), sizeof(header), 1, file);

	if (strcmp(header.filetype_identifier, "BVH") != 0) {
		printf("WARNING: BVH file '%s' has an invalid header!\n", bvh_filename);
		goto exit;
	}

	if (header.filetype_version != BVH_FILETYPE_VERSION) goto exit;

	// Check if the settings used to create the BVH file are the same as the current settings
	if (header.underlying_bvh_type    != char(BVH::underlying_bvh_type()) ||
		header.bvh_is_optimized       != config.enable_bvh_optimization ||
		header.sah_cost_node          != config.sah_cost_node ||
		header.sah_cost_leaf          != config.sah_cost_leaf
	) {
		printf("BVH file '%s' was created with different settings, rebuiling BVH from scratch.\n", bvh_filename);
		goto exit;
	}

	mesh_data.triangle_count = header.num_triangles;
	bvh.node_count           = header.num_nodes;
	bvh.index_count          = header.num_indices;

	mesh_data.triangles = new Triangle[mesh_data.triangle_count];
	bvh.nodes_2         = new BVHNode2[bvh.node_count];
	bvh.indices         = new int     [bvh.index_count];

	fread(reinterpret_cast<char *>(mesh_data.triangles), sizeof(Triangle), mesh_data.triangle_count, file);
	fread(reinterpret_cast<char *>(bvh.nodes_2),         sizeof(BVHNode2), bvh.node_count,           file);
	fread(reinterpret_cast<char *>(bvh.indices),         sizeof(int),      bvh.index_count,          file);

	printf("Loaded BVH %s from disk\n", bvh_filename);

	success = true;

exit:
	fclose(file);
	return success;
}

bool BVHLoader::save(const char * bvh_filename, const MeshData & mesh_data, const BVH & bvh) {
	FILE * file; fopen_s(&file, bvh_filename, "wb");

	if (!file) {
		printf("WARNING: Unable to save BVH to file %s!\n", bvh_filename);
		return false;
	}

	BVHFileHeader header = { };
	header.filetype_identifier[0] = 'B';
	header.filetype_identifier[1] = 'V';
	header.filetype_identifier[2] = 'H';
	header.filetype_identifier[3] = '\0';
	header.filetype_version = BVH_FILETYPE_VERSION;

	header.underlying_bvh_type    = char(BVH::underlying_bvh_type());
	header.bvh_is_optimized       = config.enable_bvh_optimization;
	header.sah_cost_node          = config.sah_cost_node;
	header.sah_cost_leaf          = config.sah_cost_leaf;

	header.num_triangles = mesh_data.triangle_count;
	header.num_nodes     = bvh.node_count;
	header.num_indices   = bvh.index_count;

	fwrite(reinterpret_cast<const char *>(&header), sizeof(header), 1, file);

	fwrite(reinterpret_cast<const char *>(mesh_data.triangles), sizeof(Triangle), mesh_data.triangle_count, file);
	fwrite(reinterpret_cast<const char *>(bvh.nodes_2),         sizeof(BVHNode2), bvh.node_count,           file);
	fwrite(reinterpret_cast<const char *>(bvh.indices),         sizeof(int),      bvh.index_count,          file);

	fclose(file);
	return true;
}
