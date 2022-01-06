#include "Sky.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <stb_image.h>

#include "Util/String.h"

void Sky::init(const String & filename) {
	int channels;
	float * hdr = stbi_loadf(filename.data(), &width, &height, &channels, STBI_rgb);

	if (!hdr || width == 0 || height == 0) {
		printf("Unable to load hdr Sky from file '%.*s'!\n", FMT_STRING(filename));
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
