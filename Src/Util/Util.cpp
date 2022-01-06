#include "Util.h"

#include <cstring>
#include <cstdio>

#include <filesystem>

static std::filesystem::path stringview_to_path(StringView str) {
	return { str.start, str.end };
}

bool Util::file_exists(StringView filename) {
	return std::filesystem::exists(stringview_to_path(filename));
}

bool Util::file_is_newer(StringView filename_a, StringView filename_b) {
	std::filesystem::file_time_type last_write_time_filename_a = std::filesystem::last_write_time(stringview_to_path(filename_a));
	std::filesystem::file_time_type last_write_time_filename_b = std::filesystem::last_write_time(stringview_to_path(filename_b));

	return last_write_time_filename_a < last_write_time_filename_b;
}

String Util::file_read(const String & filename) {
	FILE * file = nullptr;
	fopen_s(&file, filename.data(), "rb");

	if (!file) {
		printf("ERROR: Unable to open '%.*s'!\n", FMT_STRING(filename));
		abort();
	}

	// Get file length
	fseek(file, 0, SEEK_END);
	int file_length = ftell(file);
	rewind(file);

	String data(file_length);
	fread_s(data.data(), file_length, 1, file_length, file);
	data.data()[file_length] = '\0';

	fclose(file);
	return data;
}

bool Util::file_write(const String & filename, StringView data) {
	FILE * file = nullptr;
	fopen_s(&file, filename.data(), "wb");

	if (!file) return false;

	fwrite(data.start, 1, data.length(), file);
	fclose(file);

	return true;
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
	FILE * file = nullptr;
	fopen_s(&file, file_path, "wb");

	if (file == nullptr) {
		printf("Failed to export '%s'!\n", file_path);

		return;
	}

	fprintf(file, "P6\n %d\n %d\n %d\n", width, height, 255);
	fwrite(data, sizeof(char), width * height * 3, file);

	fclose(file);
}
