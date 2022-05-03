#include "BVHLoader.h"

#include <stdio.h>
#include <string.h>

#include <miniz/miniz.h>

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
	if (cpu_config.bvh_force_rebuild || !IO::file_exists(filename.view()) || !IO::file_exists(bvh_filename.view()) || IO::file_is_newer(bvh_filename.view(), filename.view())) {
		return false;
	}

	FILE * file = nullptr;
	errno_t err = fopen_s(&file, bvh_filename.c_str(), "rb");

	if (!file) {
		IO::print("WARNING: Failed to open BVH file '{}'! ({})\n"_sv, bvh_filename, IO::get_error_message(err));
		return false;
	}

	fseek(file, 0, SEEK_END);
	size_t file_size = ftell(file);
	fseek(file, 0, SEEK_SET);

	BVHFileHeader header = { };

	struct DecompressionState {
		tinfl_decompressor decompressor;

		size_t bytes_remaining_in_file;

		mz_uint8 buffer_in [32 * 1024] = { };
		mz_uint8 buffer_out[32 * 1024] = { };

		static_assert(sizeof(buffer_in)  >= TINFL_LZ_DICT_SIZE);
		static_assert(sizeof(buffer_out) >= TINFL_LZ_DICT_SIZE);

		mz_uint8 * cur_in;
		mz_uint8 * end_in;
		mz_uint8 * cur_out;
		mz_uint8 * end_out;

		DecompressionState(size_t bytes_remaining) : bytes_remaining_in_file(bytes_remaining) {
			tinfl_init(&decompressor);
			cur_in  = buffer_in;
			end_in  = buffer_in;
			cur_out = buffer_out;
			end_out = buffer_out;
		}

		NON_COPYABLE(DecompressionState);
		NON_MOVEABLE(DecompressionState);
	};
	OwnPtr<DecompressionState> state = make_owned<DecompressionState>(file_size - sizeof(BVHFileHeader));

	auto decompress_into_buffer = [&state, file](mz_uint8 * dst, size_t dst_num_bytes) -> bool {
		mz_uint8 * dst_cur = dst;
		mz_uint8 * dst_end = dst + dst_num_bytes;

		while (true) {
			size_t num_bytes_available = state->end_out - state->cur_out;
			if (dst_cur + num_bytes_available >= dst_end) {
				// There are more bytes available than we need,
				// copy over the portion that we need and don't decompress any further
				size_t num_bytes_to_copy = dst_end - dst_cur;
				memcpy(dst_cur, state->cur_out, num_bytes_to_copy);
				dst_cur       += num_bytes_to_copy;
				state->cur_out += num_bytes_to_copy;

				break;
			}
			// There are less bytes available than the total we need,
			// copy over all available bytes and start the next round of decompression
			memcpy(dst_cur, state->cur_out, num_bytes_available);
			dst_cur += num_bytes_available;
			state->cur_out = state->buffer_out;
			state->end_out = state->buffer_out;

			ASSERT(state->cur_out == state->buffer_out);
			ASSERT(state->cur_out == state->end_out);

			tinfl_status status = TINFL_STATUS_DONE;

			do {
				if (state->cur_in == state->end_in) {
					// No more input data left in buffer, go read some more from the file
					size_t num_bytes = fread_s(state->buffer_in, sizeof(state->buffer_in), sizeof(mz_uint8), sizeof(state->buffer_in), file);
					if (num_bytes == 0) {
						return false;
					}

					state->cur_in = state->buffer_in;
					state->end_in = state->buffer_in + num_bytes;

					ASSERT(state->bytes_remaining_in_file >= num_bytes);
					state->bytes_remaining_in_file -= num_bytes;
				}
				ASSERT(state->cur_in < state->end_in);

				size_t num_bytes_in  = state->end_in - state->cur_in;
				size_t num_bytes_out = state->buffer_out + sizeof(state->buffer_out) - state->cur_out;

				mz_uint32 flags = state->bytes_remaining_in_file > 0 ? TINFL_FLAG_HAS_MORE_INPUT : 0;
				status = tinfl_decompress(&state->decompressor, state->cur_in, &num_bytes_in, state->buffer_out, state->cur_out, &num_bytes_out, flags);

				bool failure = status < TINFL_STATUS_DONE;
				if (failure) {
					return false;
				}

				state->cur_in  += num_bytes_in;
				state->cur_out += num_bytes_out;
			} while (status == TINFL_STATUS_NEEDS_MORE_INPUT);

			state->end_out = state->cur_out;
			state->cur_out = state->buffer_out;
		}

		ASSERT(dst_cur == dst_end);

		return true;
	};

	bool success = false;

	size_t header_read = fread_s(&header, sizeof(header), sizeof(BVHFileHeader), 1, file);
	if (!header_read) {
		IO::print("WARNING: Failed to read header of BVH file '{}'!\n"_sv, bvh_filename);
		goto exit;
	}

	if (header.filetype_version != BVH_FILETYPE_VERSION) {
		goto exit;
	}

	// Check if the settings used to create the BVH file are the same as the current settings
	if (header.underlying_bvh_type != char(BVH::underlying_bvh_type()) ||
		header.bvh_is_optimized    != cpu_config.enable_bvh_optimization ||
		header.sah_cost_node       != cpu_config.sah_cost_node ||
		header.sah_cost_leaf       != cpu_config.sah_cost_leaf
	) {
		IO::print("BVH file '{}' was created with different settings, rebuiling BVH from scratch.\n"_sv, bvh_filename);
		goto exit;
	}

	mesh_data->triangles.resize(header.num_triangles);
	bvh->nodes          .resize(header.num_nodes);
	bvh->indices        .resize(header.num_indices);

	success =
		decompress_into_buffer(Util::bit_cast<mz_uint8 *>(mesh_data->triangles.data()), mesh_data->triangles.size() * sizeof(Triangle)) &&
		decompress_into_buffer(Util::bit_cast<mz_uint8 *>(bvh->nodes          .data()), bvh->nodes          .size() * sizeof(BVHNode2)) &&
		decompress_into_buffer(Util::bit_cast<mz_uint8 *>(bvh->indices        .data()), bvh->indices        .size() * sizeof(int));

	if (success) {
		IO::print("Loaded BVH '{}' from disk\n"_sv, bvh_filename);
	}

exit:
	fclose(file);

	return success;
}

bool BVHLoader::save(const String & bvh_filename, const MeshData & mesh_data, const BVH2 & bvh) {
	FILE * file = nullptr;
	errno_t err = fopen_s(&file, bvh_filename.data(), "wb");

	if (!file) {
		IO::print("WARNING: Failed to open BVH file '{}' for writing! ({})\n"_sv, bvh_filename, IO::get_error_message(err));
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

	tdefl_compressor compressor = { };
	bool success = false;

	size_t header_written = fwrite(&header, sizeof(header), 1, file);
	if (!header_written) {
		IO::print("WARNING: Failed to write header to BVH file '{}'!\n"_sv, bvh_filename);
		goto exit;
	}

	tdefl_put_buf_func_ptr file_append_compressed_data = [](const void * buf, int len, void * user) -> mz_bool {
		size_t bytes_written = fwrite(buf, sizeof(char), len, reinterpret_cast<FILE *>(user));
		return bytes_written == len;
	};

	constexpr static int NUM_PROBES = 256;
	tdefl_status status = tdefl_init(&compressor, file_append_compressed_data, file, NUM_PROBES);
	if (status != TDEFL_STATUS_OKAY) {
		IO::print("WARNING: Failed to initialize compressor for BVH file '{}'!\n"_sv, bvh_filename);
		goto exit;
	}

	status = tdefl_compress_buffer(&compressor, mesh_data.triangles.data(), mesh_data.triangles.size() * sizeof(Triangle), TDEFL_NO_FLUSH);
	if (status != TDEFL_STATUS_OKAY) {
		IO::print("WARNING: Failed to write compressed Triangles to BVH file '{}'!\n"_sv, bvh_filename);
		goto exit;
	}

	status = tdefl_compress_buffer(&compressor, bvh.nodes.data(), bvh.nodes.size() * sizeof(BVHNode2), TDEFL_NO_FLUSH);
	if (status != TDEFL_STATUS_OKAY) {
		IO::print("WARNING: Failed to write compressed BVH nodes to BVH file '{}'!\n"_sv, bvh_filename);
		goto exit;
	}

	status = tdefl_compress_buffer(&compressor, bvh.indices.data(), bvh.indices.size() * sizeof(int), TDEFL_FINISH);
	if (status != TDEFL_STATUS_DONE) {
		IO::print("WARNING: Failed to write compressed BVH indices to BVH file '{}'!\n"_sv, bvh_filename);
		goto exit;
	}

	success = true;

exit:
	fclose(file);
	return success;
}
