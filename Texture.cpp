#include "Texture.h"

#include <unordered_map>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include "Math.h"

static std::unordered_map<std::string, int> cache;

int     Texture::texture_count;
Texture Texture::textures[MAX_TEXTURES];

int Texture::load(const char * file_path) {
	int & texture_id = cache[file_path];

	// If the cache already contains this Texture simply return it
	if (texture_id != 0 && texture_count > 0) return texture_id;

	// Otherwise, load new Texture
	texture_id = texture_count++;
	Texture & texture = textures[texture_id];

	texture.data = stbi_load(file_path, &texture.width, &texture.height, &texture.channels, STBI_rgb_alpha);
	texture.channels = 4;
	
	// Check if the Texture is valid
	if (textures[texture_id].width == 0 || textures[texture_id].height == 0) {
		printf("An error occured while loading Texture '%s'!\n", file_path);

		abort();
	}


	return texture_id;
}
