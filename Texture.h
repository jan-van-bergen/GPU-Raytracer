#pragma once
#include "Vector3.h"

#include "Common.h"

struct Texture {
	unsigned char * data = nullptr;

	int channels;
	int width, height;
	
	static int load(const char * file_path);

	static int     texture_count;
	static Texture textures[MAX_TEXTURES];
};
