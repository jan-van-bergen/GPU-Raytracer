#include "TextureLoader.h"

#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include "Math/Mipmap.h"

bool TextureLoader::load_dds(const char * filename, Texture & texture) {
	FILE * file; fopen_s(&file, filename, "rb");

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

	texture.width  = Math::divide_round_up(texture.width,  4);
	texture.height = Math::divide_round_up(texture.height, 4);

	if (memcmp(header + 84, "DXT", 3) != 0) goto exit;

	// Get format
	// See https://en.wikipedia.org/wiki/S3_Texture_Compression
	switch (header[87]) {
		case '1': { // DXT1
			texture.format = Texture::Format::BC1;
			texture.channels = 2;
			break;
		}
		case '3': { // DXT3
			texture.format = Texture::Format::BC2;
			texture.channels = 4;
			break;
		}
		case '5': { // DXT5
			texture.format = Texture::Format::BC3;
			texture.channels = 4;
			break;
		}
		default: goto exit; // Unsupported format
	}

	int data_size = file_size - sizeof(header);

	unsigned char * data = new unsigned char[data_size];
	fread_s(data, data_size, 1, data_size, file);

	int * mip_offsets = new int[texture.mip_levels];

	int block_size = texture.channels * 4;

	int level_width  = texture.width;
	int level_height = texture.height;
	int level_offset = 0;

	for (int level = 0; level < texture.mip_levels; level++) {
		if (level_width == 0 || level_height == 0) {
			texture.mip_levels = level;
			break;
		}

		mip_offsets[level] = level_offset;
		level_offset += level_width * level_height * block_size;

		level_width  /= 2;
		level_height /= 2;
	}

	texture.data = data;
	texture.mip_offsets = mip_offsets;

	success = true;

exit:
	fclose(file);

	return success;
}

bool TextureLoader::load_stb(const char * filename, Texture & texture) {
	unsigned char * data = stbi_load(filename, &texture.width, &texture.height, &texture.channels, STBI_rgb_alpha);

	if (data == nullptr || texture.width == 0 || texture.height == 0) {
		return false;
	}

	texture.channels = 4;

#if ENABLE_MIPMAPPING
	int pixel_count = 0;

	int w = texture.width;
	int h = texture.height;

	while (true) {
		pixel_count += w * h;

		if (w == 1 && h == 1) break;

		if (w > 1) w /= 2;
		if (h > 1) h /= 2;
	}
#else
	int pixel_count = texture.width * texture.height;
#endif

	Vector4 * data_rgba = new Vector4[pixel_count];

	// Copy the data over into Mipmap level 0, and convert it to linear colour space
	for (int i = 0; i < texture.width * texture.height; i++) {
		data_rgba[i] = Vector4(
			Math::gamma_to_linear(float(data[i * 4    ]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 1]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 2]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 3]) / 255.0f)
		);
	}

	stbi_image_free(data);

#if ENABLE_MIPMAPPING
	texture.mip_levels = 1 + int(log2f(Math::max(texture.width, texture.height)));

	int * mip_offsets = new int[texture.mip_levels];
	mip_offsets[0] = 0;

	int offset      = texture.width * texture.height;
	int offset_prev = 0;

	int level_width  = texture.width  / 2;
	int level_height = texture.height / 2;

	int level = 1;

	Vector4 * temp = new Vector4[(texture.width / 2) * texture.height]; // Intermediate storage used when performing seperable filtering

	while (true) {
#if MIPMAP_DOWNSAMPLE_FILTER == MIPMAP_DOWNSAMPLE_FILTER_BOX
		// Box filter can downsample the previous Mip level
		Mipmap::downsample(level_width * 2, level_height * 2, level_width, level_height, data_rgba + offset_prev, data_rgba + offset, temp);
#else
		// Other filters downsample the original Texture for better quality
		Mipmap::downsample(texture.width, texture.height, level_width, level_height, data_rgba, data_rgba + offset, temp);
#endif

		mip_offsets[level++] = offset * sizeof(unsigned);

		if (level_width == 1 && level_height == 1) break;

		offset_prev = offset;
		offset += level_width * level_height;

		if (level_width  > 1) level_width  /= 2;
		if (level_height > 1) level_height /= 2;
	}

	delete [] temp;

	assert(level == texture.mip_levels);

	texture.mip_offsets = mip_offsets;
#else
	static constexpr int OFFSET_ZERO = 0;
	texture.mip_levels  = 1;
	texture.mip_offsets = &OFFSET_ZERO;
#endif

	// Convert floating point pixels to unsigned bytes
	unsigned * data_rgba_u8 = new unsigned[pixel_count];
	for (int i = 0; i < pixel_count; i++) {
		data_rgba_u8[i] =
			unsigned(Math::clamp(data_rgba[i].w * 255.0f, 0.0f, 255.0f)) << 24 |
			unsigned(Math::clamp(data_rgba[i].z * 255.0f, 0.0f, 255.0f)) << 16 |
			unsigned(Math::clamp(data_rgba[i].y * 255.0f, 0.0f, 255.0f)) << 8 |
			unsigned(Math::clamp(data_rgba[i].x * 255.0f, 0.0f, 255.0f));
	}
	delete [] data_rgba;

	texture.data = reinterpret_cast<const unsigned char *>(data_rgba_u8);

	return true;
}
