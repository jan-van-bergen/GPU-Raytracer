#include "Sky.h"

#include <cstdio>
#include <cstring>

#include <stb_image.h>

void Sky::init(const char * filename) {
	int channels;
	float * hdr = stbi_loadf(filename, &width, &height, &channels, STBI_rgb);

	if (!hdr || width == 0 || height == 0) {
		printf("Unable to load hdr Sky from file '%s'!\n%s\n", filename, stbi_failure_reason());
		abort();
	}

	// Allocate data and copy it over from the file
	data = new Vector3[width * height];
	memcpy(data, hdr, width * height * sizeof(Vector3));

	stbi_image_free(hdr);
}

void Sky::free() {
	delete [] data;
}
