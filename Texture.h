#pragma once
#include "Vector3.h"

struct Texture {
	unsigned char * data = nullptr;

	int channels;
	int width, height;
	
	static const Texture * load(const char * file_path);
};
