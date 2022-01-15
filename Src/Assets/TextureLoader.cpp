#include "TextureLoader.h"

#include <ctype.h>
#include <string.h>

#define STB_DXT_IMPLEMENTATION
#include <stb_dxt.h>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "Config.h"

#include "Core/Parser.h"

#include "Math/Mipmap.h"
#include "Util/Util.h"

bool TextureLoader::load_dds(const String & filename, Texture & texture) {
	String file = IO::file_read(filename);
	Parser parser(file.view(), filename.view());

	// Based on: https://docs.microsoft.com/en-us/windows/win32/direct3ddds/dds-header
	struct DDSHeader {
		char     identifier[4];
		unsigned size;
		unsigned flags;
		unsigned height;
		unsigned width;
		unsigned pitch;
		unsigned depth;
		unsigned num_mipmaps;
		unsigned reserved1[11];
		struct DDS_PIXELFORMAT {
			unsigned size;
			unsigned flags;
			char     four_cc[4];
			unsigned rgb_bitcount;
			unsigned bimask_r;
			unsigned bimask_g;
			unsigned bimask_b;
			unsigned bimask_a;
		} spf;
		unsigned caps;
		unsigned caps_2;
		unsigned caps_3;
		unsigned caps_4;
		unsigned reserved_2;
	};
	static_assert(sizeof(DDSHeader) == 128);

	DDSHeader header = parser.parse_binary<DDSHeader>();

	// First four bytes should be "DDS "
	if (memcmp(header.identifier, "DDS ", 4) != 0) return false;

	texture.width  = Math::divide_round_up(header.width,  4u);
	texture.height = Math::divide_round_up(header.height, 4u);

	if (memcmp(header.spf.four_cc, "DXT", 3) != 0) return false;

	// Get format
	switch (header.spf.four_cc[3]) {
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
		default: return false;
	}

	size_t data_size = file.size() - sizeof(header);

	texture.data.resize(data_size);
	memcpy(texture.data.data(), parser.cur, data_size);

	int block_size = texture.channels * 4;

	int level_width  = texture.width;
	int level_height = texture.height;
	int level_offset = 0;

	for (unsigned level = 0; level < header.num_mipmaps; level++) {
		if (level_width == 0 || level_height == 0) {
			break;
		}

		texture.mip_offsets.push_back(level_offset);
		level_offset += level_width * level_height * block_size;

		level_width  /= 2;
		level_height /= 2;
	}

	return true;
}

static void mip_count(int width, int height, int & mip_levels, int & pixel_count) {
	if (config.enable_mipmapping) {
		mip_levels  = 0;
		pixel_count = 0;

		while (true) {
			mip_levels++;
			pixel_count += width * height;

			if (width == 1 && height == 1) break;

			if (width  > 1) width  /= 2;
			if (height > 1) height /= 2;
		}
	} else {
		mip_levels  = 1;
		pixel_count = width * height;
	}
}

bool TextureLoader::load_stb(const String & filename, Texture & texture) {
	unsigned char * data = stbi_load(filename.data(), &texture.width, &texture.height, &texture.channels, STBI_rgb_alpha);

	if (data == nullptr || texture.width == 0 || texture.height == 0) {
		return false;
	}

	texture.channels = 4;

	int mip_levels  = 0;
	int pixel_count = 0;
	mip_count(texture.width, texture.height, mip_levels, pixel_count);

	Array<Vector4> data_rgba(pixel_count);

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

	texture.mip_offsets.push_back(0);

	if (config.enable_mipmapping) {
		int offset      = texture.width * texture.height;
		int offset_prev = 0;

		int level_width_prev  = texture.width;
		int level_height_prev = texture.height;

		int level_width  = texture.width  / 2;
		int level_height = texture.height / 2;

		int level = 1;

		Array<Vector4> temp((texture.width / 2) * texture.height); // Intermediate storage used when performing seperable filtering

		while (true) {
			if (config.mipmap_filter == Config::MipmapFilter::BOX) {
				// Box filter can downsample the previous Mip level
				Mipmap::downsample(level_width_prev, level_height_prev, level_width, level_height, data_rgba.data() + offset_prev, data_rgba.data() + offset, temp.data());
			} else {
				// Other filters downsample the original Texture for better quality
				Mipmap::downsample(texture.width, texture.height, level_width, level_height, data_rgba.data(), data_rgba.data() + offset, temp.data());
			}

			texture.mip_offsets.push_back(offset * sizeof(unsigned));

			if (level_width == 1 && level_height == 1) break;

			offset_prev = offset;
			offset += level_width * level_height;

			level_width_prev  = level_width;
			level_height_prev = level_height;

			if (level_width  > 1) level_width  /= 2;
			if (level_height > 1) level_height /= 2;
		}

		ASSERT(texture.mip_offsets.size() == mip_levels);
	}

	// Convert floating point pixels to unsigned bytes
	Array<unsigned char> data_rgba_u8(pixel_count * 4);
	for (int i = 0; i < pixel_count; i++) {
		data_rgba_u8[4*i + 0] = unsigned char(Math::clamp(data_rgba[i].x * 255.0f, 0.0f, 255.0f));
		data_rgba_u8[4*i + 1] = unsigned char(Math::clamp(data_rgba[i].y * 255.0f, 0.0f, 255.0f));
		data_rgba_u8[4*i + 2] = unsigned char(Math::clamp(data_rgba[i].z * 255.0f, 0.0f, 255.0f));
		data_rgba_u8[4*i + 3] = unsigned char(Math::clamp(data_rgba[i].w * 255.0f, 0.0f, 255.0f));
	}

	if (config.enable_block_compression && Math::is_power_of_two(texture.width) && Math::is_power_of_two(texture.height)) {
		// Block Compression
		int new_width  = Math::divide_round_up(texture.width,  4);
		int new_height = Math::divide_round_up(texture.height, 4);

		int new_mip_levels  = 0;
		int new_pixel_count = 0;
		mip_count(new_width, new_height, new_mip_levels, new_pixel_count);

		Array<int> new_mip_offsets;

		constexpr int COMPRESSED_BLOCK_SIZE = 8;

		Array<unsigned char> compressed_data(new_pixel_count * COMPRESSED_BLOCK_SIZE);
		int                  compressed_data_offset = 0;

		for (int l = 0; l < new_mip_levels; l++) {
			new_mip_offsets.push_back(compressed_data_offset);

			unsigned char * level_data = data_rgba_u8.data() + texture.mip_offsets[l];

			int level_width  = Math::max(texture.width  >> l, 1);
			int level_height = Math::max(texture.height >> l, 1);

			int new_level_width  = Math::max(new_width  >> l, 1);
			int new_level_height = Math::max(new_height >> l, 1);

			for (int y = 0; y < new_level_height; y++) {
				for (int x = 0; x < new_level_width; x++) {
					unsigned char block[4 * 4 * 4] = { };

					for (int j = 0; j < 4; j++) {
						int pixel_y = 4*y + j;
						if (pixel_y < level_height) {
							for (int i = 0; i < 4; i++) {
								int pixel_x = 4*x + i;
								if (pixel_x < level_width) {
									int block_index = i + j * 4;
									int pixel_index = pixel_x + pixel_y * level_width;

									block[4*block_index + 0] = level_data[4*pixel_index + 0];
									block[4*block_index + 1] = level_data[4*pixel_index + 1];
									block[4*block_index + 2] = level_data[4*pixel_index + 2];
									block[4*block_index + 3] = level_data[4*pixel_index + 3];
								}
							}
						}
					}

					unsigned char compressed_block[COMPRESSED_BLOCK_SIZE] = { };
					stb_compress_dxt_block(compressed_block, block, false, STB_DXT_HIGHQUAL);

					for (int i = 0; i < COMPRESSED_BLOCK_SIZE; i++) {
						compressed_data[compressed_data_offset++] = compressed_block[i];
					}
				}
			}
		}

		ASSERT(compressed_data_offset == new_pixel_count * COMPRESSED_BLOCK_SIZE);
		data_rgba_u8 = std::move(compressed_data);

		texture.format   = Texture::Format::BC1;
		texture.channels = 2;
		texture.width    = new_width;
		texture.height   = new_height;

		texture.mip_offsets = std::move(new_mip_offsets);
	}

	texture.data = std::move(data_rgba_u8);

	return true;
}
