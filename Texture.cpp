#include "Texture.h"

#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include "Math.h"

static std::unordered_map<std::string, int> cache;

std::vector<Texture> Texture::textures;

int Texture::load(const char * file_path) {
	int & texture_id = cache[file_path];

	// If the cache already contains this Texture simply return it
	if (texture_id != 0 && textures.size() > 0) return texture_id;

	// Otherwise, create new Texture and load it from disk
	texture_id        = textures.size();
	Texture & texture = textures.emplace_back();

	texture.data = stbi_load(file_path, &texture.width, &texture.height, &texture.channels, STBI_rgb_alpha);
	texture.channels = 4;
	
	// Check if the Texture is valid
	if (texture.data == nullptr || texture.width == 0 || texture.height == 0) {
		printf("WARNING: Failed to load Texture '%s'!\n", file_path);

		texture.width  = 1;
		texture.height = 1;
		texture.data = new unsigned char[4] { 255, 0, 255, 255 };
	}

	return texture_id;
}

void Texture::shrink() {
	if (width == 1 || height == 1) abort();

	int new_width  = width  >> 1;
	int new_height = height >> 1;

	unsigned char * new_data = new unsigned char[new_width * new_height * channels];

	for (int j = 0; j < new_height; j++) {
		for (int i = 0; i < new_width; i++) {
			int x = i << 1;
			int y = j << 1;

			int index_0 = ((x)   + (y)   * height) << 2;
			int index_1 = ((x+1) + (y)   * height) << 2;
			int index_2 = ((x)   + (y+1) * height) << 2;
			int index_3 = ((x+1) + (y+1) * height) << 2;

			unsigned r = data[index_0    ] + data[index_1    ] + data[index_2    ] + data[index_3    ];
			unsigned g = data[index_0 + 1] + data[index_1 + 1] + data[index_2 + 1] + data[index_3 + 1];
			unsigned b = data[index_0 + 2] + data[index_1 + 2] + data[index_2 + 2] + data[index_3 + 2];
			unsigned a = data[index_0 + 3] + data[index_1 + 3] + data[index_2 + 3] + data[index_3 + 3];

			int new_index = (i + j * new_height) << 2;
			new_data[new_index    ] = r >> 2;
			new_data[new_index + 1] = g >> 2;
			new_data[new_index + 2] = b >> 2;
			new_data[new_index + 3] = a >> 2;
		}
	}

	delete [] data;

	width  = new_width;
	height = new_height;
	data   = new_data;
}