#include "Sky.h"

#include <cstdio>
#include <stdlib.h>

#include "Util.h"

void Sky::init(const char * file_path) {
	FILE * file; 
	fopen_s(&file, file_path, "rb");

	if (!file) abort();

	// Seek to the end to obtain the total pixel count
	fseek(file, 0, SEEK_END);
	int size_squared = ftell(file) / sizeof(Vector3);
	rewind(file);

	// The image is square, so take a square root to obtain the side lengths
	size = int(sqrtf(size_squared));
	assert(size * size == size_squared);

	// Allocate data and copy it over from the file
	data = new Vector3[size_squared];
	fread(reinterpret_cast<char *>(data), sizeof(Vector3), size_squared, file);
	fclose(file);
}
