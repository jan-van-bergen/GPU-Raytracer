#include "BVHLoader.h"

#include <stdio.h>
#include <string.h>

#include "Util/Util.h"
#include "Util/IO.h"
#include "Util/Parser.h"
#include "Util/StringUtil.h"

String BVHLoader::get_bvh_filename(StringView filename) {
	return Util::combine_stringviews(filename, StringView::from_c_str(BVH_FILE_EXTENSION));
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

bool BVHLoader::try_to_load(const String & filename, const String & bvh_filename, MeshData & mesh_data, BVH & bvh) {
	if (config.bvh_force_rebuild || !IO::file_exists(bvh_filename.view()) || IO::file_is_newer(bvh_filename.view(), filename.view())) {
		return false;
	}

	String file = IO::file_read(bvh_filename);

	Parser parser = { };
	parser.init(file.view(), bvh_filename.view());

	BVHFileHeader header = parser.parse_binary<BVHFileHeader>();

	if (strcmp(header.filetype_identifier, "BVH") != 0) {
		IO::print("WARNING: BVH file '{}' has an invalid header!\n"sv, bvh_filename);
		return false;
	}

	if (header.filetype_version != BVH_FILETYPE_VERSION) return false;

	// Check if the settings used to create the BVH file are the same as the current settings
	if (header.underlying_bvh_type    != char(BVH::underlying_bvh_type()) ||
		header.bvh_is_optimized       != config.enable_bvh_optimization ||
		header.sah_cost_node          != config.sah_cost_node ||
		header.sah_cost_leaf          != config.sah_cost_leaf
	) {
		IO::print("BVH file '{}' was created with different settings, rebuiling BVH from scratch.\n"sv, bvh_filename);
		return false;
	}

	mesh_data.triangle_count = header.num_triangles;
	bvh.node_count           = header.num_nodes;
	bvh.index_count          = header.num_indices;

	mesh_data.triangles = new Triangle[mesh_data.triangle_count];
	bvh.nodes._2        = new BVHNode2[bvh.node_count];
	bvh.indices         = new int     [bvh.index_count];

	for (int i = 0; i < mesh_data.triangle_count; i++) mesh_data.triangles[i] = parser.parse_binary<Triangle>();
	for (int i = 0; i < bvh.node_count;           i++) bvh.nodes._2       [i] = parser.parse_binary<BVHNode2>();
	for (int i = 0; i < bvh.index_count;          i++) bvh.indices        [i] = parser.parse_binary<int>();

	ASSERT(parser.reached_end());

	IO::print("Loaded BVH '{}' from disk\n"sv, bvh_filename);
	return true;
}

bool BVHLoader::save(const String & bvh_filename, const MeshData & mesh_data, const BVH & bvh) {
	FILE * file = nullptr;
	fopen_s(&file, bvh_filename.data(), "wb");

	if (!file) {
		IO::print("WARNING: Unable to open BVH file '{}' for writing!\n"sv, bvh_filename);
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

	size_t header_written = fwrite(reinterpret_cast<const char *>(&header), sizeof(header), 1, file);

	size_t num_triangles_written = fwrite(reinterpret_cast<const char *>(mesh_data.triangles), sizeof(Triangle), mesh_data.triangle_count, file);
	size_t num_bvh_nodes_written = fwrite(reinterpret_cast<const char *>(bvh.nodes._2),        sizeof(BVHNode2), bvh.node_count,           file);
	size_t num_indices_written   = fwrite(reinterpret_cast<const char *>(bvh.indices),         sizeof(int),      bvh.index_count,          file);

	fclose(file);

	if (!header_written || num_triangles_written < mesh_data.triangle_count || num_bvh_nodes_written < bvh.node_count || num_indices_written < bvh.index_count) {
		IO::print("WARNING: Unable to successfully write to BVH file '{}'!\n"sv, bvh_filename);
		return false;
	}

	return true;
}
