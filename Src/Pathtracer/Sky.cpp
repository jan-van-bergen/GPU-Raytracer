#include "Sky.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <stb_image.h>

#include "Core/IO.h"
#include "Core/String.h"

void Sky::load(const String & filename) {
	int channels;
	float * hdr = stbi_loadf(filename.data(), &width, &height, &channels, STBI_rgb);

	if (!hdr || width == 0 || height == 0) {
		IO::print("Unable to load hdr Sky from file '{}'!\n"sv, filename);
		abort();
	}

	// Allocate data and copy it over from the file
	data.resize(width * height);
	memcpy(data.data(), hdr, width * height * sizeof(Vector3));

	stbi_image_free(hdr);
}
