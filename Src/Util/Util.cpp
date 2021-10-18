#include "Util.h"

#include <cstring>
#include <cstdio>

#include <filesystem>

void Util::get_path(const char * filename, char * path) {
	const char * path_end      = filename;
	const char * last_path_end = nullptr;

	// Keep advancing the path_end pointer until we run out of '/' or '\\' characters in the string
	while (path_end = strpbrk(path_end, "/\\")) {
		path_end++;
		last_path_end = path_end;
	}

	if (last_path_end == nullptr) {
		path[0] = NULL;

		return;
	}

	// Copy the right amount over
	int path_length = last_path_end - filename;
	memcpy(path, filename, path_length);
	path[path_length] = NULL;
}

bool Util::file_exists(const char * filename) {
	return std::filesystem::exists(filename);
}

bool Util::file_is_newer(const char * file_reference, const char * file_check) {
	std::filesystem::file_time_type last_write_time_reference = std::filesystem::last_write_time(file_reference);
	std::filesystem::file_time_type last_write_time_check     = std::filesystem::last_write_time(file_check);

	return last_write_time_reference < last_write_time_check;
}

char * Util::file_read(const char * filename, int & file_length) {
	FILE * file;
	fopen_s(&file, filename, "rb");

	if (file == nullptr) {
		printf("ERROR: Unable to open '%s'!\n", filename);
		abort();
	}

	// Get file length
	fseek(file, 0, SEEK_END);
	file_length = ftell(file);
	rewind(file);

	// Copy file source into c string
	char * data = new char[file_length + 1];
	fread_s(data, file_length + 1, 1, file_length, file);

	fclose(file);

	data[file_length] = NULL;
	return data;
}

const char * Util::find_last(const char * haystack, const char * needles) {
	const char * ptr_prev = nullptr;
	const char * ptr_curr = haystack;

	while (true) {
		ptr_curr = strpbrk(ptr_curr, needles);

		if (ptr_curr == nullptr) return ptr_prev;

		ptr_prev = ++ptr_curr;
	}
}

// Based on: Vose - A Linear Algorithm for Generating Random Numbers with a Given Distribution (1991)
void Util::init_alias_method(int n, double p[], ProbAlias distribution[]) {
	int * large = new int[n];
	int * small = new int[n];
	int   l = 0;
	int   s = 0;

	for (int j = 0; j < n; j++) {
		p[j] *= double(n);
		if (p[j] < 1.0) {
			small[s++] = j;
		} else {
			large[l++] = j;
		}
	}

	while (s != 0 && l != 0) {
		int j = small[--s];
		int k = large[--l];

		distribution[j].prob  = p[j];
		distribution[j].alias = k;

		p[k] = (p[k] + p[j]) - 1.0;

		if (p[k] < 1.0) {
			small[s++] = k;
		} else {
			large[l++] = k;
		}
	}

	while (s > 0) distribution[small[--s]] = { 1.0f, -1 };
	while (l > 0) distribution[large[--l]] = { 1.0f, -1 };

	delete [] large;
	delete [] small;
}

// Based on: https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file
void Util::export_ppm(const char * file_path, int width, int height, const unsigned char * data) {
	FILE * file;
	fopen_s(&file, file_path, "wb");

	if (file == nullptr) {
		printf("Failed to export '%s'!\n", file_path);

		return;
	}

	fprintf(file, "P6\n %d\n %d\n %d\n", width, height, 255);
	fwrite(data, sizeof(char), width * height * 3, file);

	fclose(file);
}
