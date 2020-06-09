#include "Texture.h"

#include <unordered_map>
#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include "Math.h"

static std::unordered_map<std::string, int> cache;

std::vector<Texture> Texture::textures;

static bool load_dds(Texture & texture, const char * file_path) {
	FILE * file;
	fopen_s(&file, file_path, "rb");

	if (file == nullptr) return false;

	unsigned char header[128];
	fread_s(header, sizeof(header), 1, 128, file);

	// First four bytes should be "DDS "
	if (memcmp(header, "DDS ", 4) != 0) return false;

	// Get width and height
	memcpy_s(&texture.width,  sizeof(texture.width),  header + 16, sizeof(int));
	memcpy_s(&texture.height, sizeof(texture.height), header + 12, sizeof(int));

	texture.width  = (texture.width  + 3) >> 2;
	texture.height = (texture.height + 3) >> 2;

	if (memcmp(header + 84, "DXT", 3) != 0) return false;

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

		default: return false; // Unsupported
	}

	int data_size = texture.get_width_in_bytes() * texture.height;

	texture.data = new unsigned char[data_size];
	fread_s(texture.data, data_size, 1, data_size, file);

	return true;
}

static bool load_stbi(Texture & texture, const char * file_path) {
	texture.data = stbi_load(file_path, &texture.width, &texture.height, &texture.channels, STBI_rgb_alpha);
	texture.channels = 4;
	
	// Check if the Texture is valid
	return texture.data != nullptr && texture.width > 0 && texture.height > 0;
}

int Texture::load(const char * file_path) {
	int & texture_id = cache[file_path];

	// If the cache already contains this Texture simply return it
	if (texture_id != 0 && textures.size() > 0) return texture_id;
	
	// Otherwise, create new Texture and load it from disk
	texture_id        = textures.size();
	Texture & texture = textures.emplace_back();

	int    file_path_length = strlen(file_path);
	char * file_extension = nullptr;

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

	if (file_extension == nullptr) abort();

	bool success;
	if (strcmp(file_extension, "dds") == 0) {
		success = load_dds(texture, file_path); // DDS is loaded using custom code
	} else {
		success = load_stbi(texture, file_path); // other file formats use stb_image
	}

	if (!success) {
		printf("WARNING: Failed to load Texture '%s'!\n", file_path);

		if (texture.data) delete [] texture.data;

		// Make Texture pure pink to signify invalid Texture
		texture.format = Texture::Format::RGBA;
		texture.width  = 1;
		texture.height = 1;
		texture.channels = 4;
		texture.data = new unsigned char[4] { 255, 0, 255, 255 };
	}

	delete [] file_extension;

	return texture_id;
}

CUarray_format Texture::get_cuda_array_format() const {
	switch (format) {
		case Format::RGBA_COMPRESSED_BC1: return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::RGBA_COMPRESSED_BC2: return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::RGBA_COMPRESSED_BC3: return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::RGBA:                return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8;
	}
}

CUresourceViewFormat Texture::get_cuda_resource_view_format() const {
	switch (format) {
		case Texture::Format::RGBA_COMPRESSED_BC1: return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
		case Texture::Format::RGBA_COMPRESSED_BC2: return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
		case Texture::Format::RGBA_COMPRESSED_BC3: return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
		case Texture::Format::RGBA:                return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UINT_4X8;
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
		return width * channels;
	} else {
		return width * channels * sizeof(int);
	}
}
