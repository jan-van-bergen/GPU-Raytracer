#pragma once
#include <vector>

#include "Common.h"

struct Texture {
	unsigned char * data = nullptr;

	int channels;
	int width, height;
	
	void shrink();

	static int load(const char * file_path);

	static std::vector<Texture> textures;
};
