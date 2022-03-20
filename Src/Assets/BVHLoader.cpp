#include "BVHLoader.h"

#include <stdio.h>
#include <string.h>

#include "Core/IO.h"
#include "Core/Parser.h"
#include "Core/Allocators/StackAllocator.h"

#include "Util/Util.h"
#include "Util/StringUtil.h"

String BVHLoader::get_bvh_filename(StringView filename, Allocator * allocator) {
	return Util::combine_stringviews(filename, StringView::from_c_str(BVH_FILE_EXTENSION), allocator);
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

bool BVHLoader::try_to_load(const String & filename, const String & bvh_filename, MeshData * mesh_data, BVH2 * bvh) {
	if (cpu_config.bvh_force_rebuild || !IO::file_exists(bvh_filename.view()) || IO::file_is_newer(bvh_filename.view(), filename.view())) {
		return false;
	}

	StackAllocator<KILOBYTES(8)> allocator;
	String file = IO::file_read(bvh_filename, &allocator);
	Parser parser(file.view(), bvh_filename.view());

	BVHFileHeader header = parser.parse_binary<BVHFileHeader>();

	if (strcmp(header.filetype_identifier, "BVH") != 0) {
		IO::print("WARNING: BVH file '{}' has an invalid header!\n"_sv, bvh_filename);
		return false;
	}

	if (header.filetype_version != BVH_FILETYPE_VERSION) return false;

	// Check if the settings used to create the BVH file are the same as the current settings
	if (header.underlying_bvh_type != char(BVH::underlying_bvh_type()) ||
		header.bvh_is_optimized    != cpu_config.enable_bvh_optimization ||
		header.sah_cost_node       != cpu_config.sah_cost_node ||
		header.sah_cost_leaf       != cpu_config.sah_cost_leaf
	) {
		IO::print("BVH file '{}' was created with different settings, rebuiling BVH from scratch.\n"_sv, bvh_filename);
		return false;
	}

	mesh_data->triangles.resize(header.num_triangles);
	bvh->nodes          .resize(header.num_nodes);
	bvh->indices        .resize(header.num_indices);

	parser.copy_into(mesh_data->triangles.data(), mesh_data->triangles.size());
	parser.copy_into(bvh->nodes  .data(), bvh->nodes  .size());
	parser.copy_into(bvh->indices.data(), bvh->indices.size());

	ASSERT(parser.reached_end());

	IO::print("Loaded BVH '{}' from disk\n"_sv, bvh_filename);
	return true;
}

bool BVHLoader::save(const String & bvh_filename, const MeshData & mesh_data, const BVH2 & bvh) {
	FILE * file = nullptr;
	fopen_s(&file, bvh_filename.data(), "wb");

	if (!file) {
		IO::print("WARNING: Unable to open BVH file '{}' for writing!\n"_sv, bvh_filename);
		return false;
	}

	BVHFileHeader header = { };
	header.filetype_identifier[0] = 'B';
	header.filetype_identifier[1] = 'V';
	header.filetype_identifier[2] = 'H';
	header.filetype_identifier[3] = '\0';
	header.filetype_version = BVH_FILETYPE_VERSION;

	header.underlying_bvh_type = char(BVH::underlying_bvh_type());
	header.bvh_is_optimized    = cpu_config.enable_bvh_optimization;
	header.sah_cost_node       = cpu_config.sah_cost_node;
	header.sah_cost_leaf       = cpu_config.sah_cost_leaf;

	header.num_triangles = mesh_data.triangles.size();
	header.num_nodes     = bvh.nodes  .size();
	header.num_indices   = bvh.indices.size();

	size_t header_written = fwrite(&header, sizeof(header), 1, file);

	size_t num_triangles_written = fwrite(mesh_data.triangles.data(), sizeof(Triangle), mesh_data.triangles.size(), file);
	size_t num_bvh_nodes_written = fwrite(bvh.nodes  .data(),         sizeof(BVHNode2), bvh.nodes  .size(),         file);
	size_t num_indices_written   = fwrite(bvh.indices.data(),         sizeof(int),      bvh.indices.size(),         file);

	fclose(file);

	if (!header_written || num_triangles_written < mesh_data.triangles.size() || num_bvh_nodes_written < bvh.nodes.size() || num_indices_written < bvh.indices.size()) {
		IO::print("WARNING: Unable to successfully write to BVH file '{}'!\n"_sv, bvh_filename);
		return false;
	}

	return true;
}
