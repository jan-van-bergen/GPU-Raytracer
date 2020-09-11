#include "Texture.h"

#include <unordered_map>
#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include "Math.h"

#include "Vector4.h"

static std::unordered_map<std::string, int> cache;

static bool load_dds(Texture & texture, const char * file_path) {
	FILE * file;
	fopen_s(&file, file_path, "rb");

	if (file == nullptr) return false;
	
	bool success = false;

	fseek(file, 0, SEEK_END);
	int file_size = ftell(file);
	fseek(file, 0, SEEK_SET);

	unsigned char header[128];
	fread_s(header, sizeof(header), 1, 128, file);

	// First four bytes should be "DDS "
	if (memcmp(header, "DDS ", 4) != 0) goto exit;

	// Get width and height
	memcpy_s(&texture.width,      sizeof(int), header + 16, sizeof(int));
	memcpy_s(&texture.height,     sizeof(int), header + 12, sizeof(int));
	memcpy_s(&texture.mip_levels, sizeof(int), header + 28, sizeof(int));

	texture.width  = (texture.width  + 3) / 4;
	texture.height = (texture.height + 3) / 4;

	texture.mip_offsets = new int[texture.mip_levels];
	texture.mip_offsets[0] = 0;

	if (memcmp(header + 84, "DXT", 3) != 0) goto exit;

	// Get format
	// See https://en.wikipedia.org/wiki/S3_Texture_Compression
	switch (header[87]) {
		case '1': { // DXT1
			texture.format = Texture::Format::RGBA_COMPRESSED_BC1;
			texture.channels = 2;

			break;
		}
		case '3': { // DXT3
			texture.format = Texture::Format::RGBA_COMPRESSED_BC2;
			texture.channels = 4;

			break;
		}
		case '5': { // DXT5
			texture.format = Texture::Format::RGBA_COMPRESSED_BC3;
			texture.channels = 4;

			break;
		}

		default: goto exit; // Unsupported format
	}

	int data_size = file_size - sizeof(header);

	unsigned char * data = new unsigned char[data_size];
	fread_s(data, data_size, 1, data_size, file);

	texture.data = data;

	int block_size = texture.channels * 4;

	int level_width  = texture.width;
	int level_height = texture.height;
	int level_offset = 0;

	for (int level = 0; level < texture.mip_levels; level++) {
		if (level_width == 0 || level_height == 0) {
			texture.mip_levels = level;
			
			break;
		}

		texture.mip_offsets[level] = level_offset;
		level_offset += level_width * level_height * block_size;

		level_width  /= 2;
		level_height /= 2;
	}

	success = true;

exit:
	fclose(file);

	return success;
}

static bool load_stbi(Texture & texture, const char * file_path) {
	unsigned char * data = stbi_load(file_path, &texture.width, &texture.height, &texture.channels, STBI_rgb_alpha);

	if (data == nullptr || texture.width == 0 || texture.height == 0) return false;

	texture.channels = 4;
	
	int size = texture.width * texture.height;
	Vector4 * data_rgba = new Vector4[size + size / 3];

	// Copy the data over into Mipmap level 0, and convert it to linear colour space
	for (int i = 0; i < texture.width * texture.height; i++) {
		data_rgba[i].x = Math::gamma_to_linear(data[i * 4    ] / 255.0f);
		data_rgba[i].y = Math::gamma_to_linear(data[i * 4 + 1] / 255.0f);
		data_rgba[i].z = Math::gamma_to_linear(data[i * 4 + 2] / 255.0f);
		data_rgba[i].w = Math::gamma_to_linear(data[i * 4 + 3] / 255.0f);
	}

	texture.mip_levels  = 1 + int(log2f(Math::max(texture.width, texture.height)));
	texture.mip_offsets = new int[texture.mip_levels];
	texture.mip_offsets[0] = 0;

	int offset      = texture.width * texture.height;
	int offset_prev = 0;

	int level_width       = texture.width  / 2;
	int level_height      = texture.height / 2;
	int level_width_prev  = texture.width;
	int level_height_prev = texture.height;
	
	int level = 1;

	// Obtain each subsequent Mipmap level by applying a Box Filter to the previous level
	while (level_width >= 1 || level_height >= 1) {
		Vector4 const * mip_level_prev = data_rgba + offset_prev;
		Vector4       * mip_level      = data_rgba + offset;

		for (int j = 0; j < level_height; j++) {
			for (int i = 0; i < level_width; i++) {
				int i_prev = i * 2;
				int j_prev = j * 2;

				// Box filter
				Vector4 colour_0 = mip_level_prev[(i_prev)     + (j_prev)     * level_width_prev];
				Vector4 colour_1 = mip_level_prev[(i_prev + 1) + (j_prev)     * level_width_prev];
				Vector4 colour_2 = mip_level_prev[(i_prev)     + (j_prev + 1) * level_width_prev];
				Vector4 colour_3 = mip_level_prev[(i_prev + 1) + (j_prev + 1) * level_width_prev];

				mip_level[i + j * level_width] = (colour_0 + colour_1 + colour_2 + colour_3) * 0.25f;
			}
		}

		texture.mip_offsets[level++] = offset * sizeof(Vector4);

		offset_prev = offset;
		offset += level_width * level_height;

		level_width_prev  = level_width;
		level_height_prev = level_height;
		level_width  /= 2;
		level_height /= 2;
	}

	assert(level == texture.mip_levels);

	texture.data = reinterpret_cast<const unsigned char *>(data_rgba);

	return true;
}

int Texture::load(const char * file_path) {
	int & texture_id = cache[file_path];

	// If the cache already contains this Texture simply return it
	if (texture_id != 0 && textures.size() > 0) return texture_id;
	
	// Otherwise, create new Texture and load it from disk
	texture_id        = textures.size();
	Texture & texture = textures.emplace_back();

	int    file_path_length = strlen(file_path);
	char * file_extension   = nullptr;

	// Extract file extension from path
	for (int i = file_path_length - 1; i >= 0; i--) {
		if (file_path[i] == '.') {
			int file_extension_length = file_path_length - i;
			file_extension = new char[file_extension_length];

			for (int j = 0; j < file_extension_length; j++) {
				file_extension[j] = tolower(file_path[i + 1 + j]);
			}

			break;
		}
	}

	bool success = false;

	if (file_extension) {
		if (strcmp(file_extension, "dds") == 0) {
			success = load_dds(texture, file_path); // DDS is loaded using custom code
		} else {
			success = load_stbi(texture, file_path); // other file formats use stb_image
		}
	}

	if (!success) {
		printf("WARNING: Failed to load Texture '%s'!\n", file_path);

		if (texture.data) delete [] texture.data;

		static int zero = 0;
		static const Vector4 pink = Vector4(1.0f, 0, 1.0f, 1.0f);

		// Make Texture pure pink to signify invalid Texture
		texture.data = reinterpret_cast<const unsigned char *>(&pink);
		texture.format = Texture::Format::RGBA;
		texture.width  = 1;
		texture.height = 1;
		texture.channels = 4;
		texture.mip_levels  = 1;
		texture.mip_offsets = &zero;
	}

	delete [] file_extension;

	return texture_id;
}

CUarray_format Texture::get_cuda_array_format() const {
	switch (format) {
		case Format::RGBA_COMPRESSED_BC1: return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::RGBA_COMPRESSED_BC2: return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::RGBA_COMPRESSED_BC3: return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::RGBA:                return CUarray_format::CU_AD_FORMAT_FLOAT;
	}
}

CUresourceViewFormat Texture::get_cuda_resource_view_format() const {
	switch (format) {
		case Texture::Format::RGBA_COMPRESSED_BC1: return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
		case Texture::Format::RGBA_COMPRESSED_BC2: return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
		case Texture::Format::RGBA_COMPRESSED_BC3: return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
		case Texture::Format::RGBA:                return CUresourceViewFormat::CU_RES_VIEW_FORMAT_FLOAT_4X32;
	}
}

int Texture::get_cuda_resource_view_width() const {
	if (format == Format::RGBA) {
		return width;
	} else {
		return width * 4;
	}
}

int Texture::get_cuda_resource_view_height() const {
	if (format == Format::RGBA) {
		return height;
	} else {
		return height * 4;
	}
}

int Texture::get_width_in_bytes() const {
	if (format == Format::RGBA) {
		return width * sizeof(Vector4);
	} else {
		return width * channels * 4;
	}
}
