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
		IO::print("Unable to load hdr Sky from file '{}'!\n"_sv, filename);
		IO::exit(1);
	}

	// Allocate data and copy it over from the file
	data.resize(width * height);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = x + y * width;
			data[index] = Vector4(
				hdr[3*index + 0],
				hdr[3*index + 1],
				hdr[3*index + 2],
				0.0f
			);
		}
	}

	stbi_image_free(hdr);
}
