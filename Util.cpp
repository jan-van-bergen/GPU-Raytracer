#include "Util.h"

#include <cstring>
#include <cstdio>

#include <filesystem>

const char * Util::get_path(const char * file_path) {
	const char * path_end      = file_path;
	const char * last_path_end = nullptr;

	// Keep advancing the path_end pointer until we run out of '/' characters in the string
	while (path_end = strchr(path_end, '/')) {
		path_end++;
		last_path_end = path_end;
	}

	if (last_path_end == nullptr) return nullptr;

	char * path = new char[strlen(file_path)];

	// Copy the right amount over
	int path_length = last_path_end - file_path;
	memcpy(path, file_path, path_length);
	path[path_length] = NULL;

	return path;
}

bool Util::file_exists(const char * filename) {
	return std::filesystem::exists(filename);
}

bool Util::file_is_newer(const char * file_reference, const char * file_check) {
	std::filesystem::file_time_type last_write_time_reference = std::filesystem::last_write_time(file_reference);
	std::filesystem::file_time_type last_write_time_check     = std::filesystem::last_write_time(file_check);

	return last_write_time_reference < last_write_time_check;
}

// Based on: https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file
void Util::export_ppm(const char * file_path, int width, int height, const unsigned char * data) {
	FILE * file;
	fopen_s(&file, file_path, "wb");

	if (file == nullptr) {
		printf("Failed to take export %s!\n", file_path);

		return;
	}

	fprintf(file, "P6\n %d\n %d\n %d\n", width, height, 255);
	fwrite(data, sizeof(char), width * height * 3, file);

	fclose(file);
}
