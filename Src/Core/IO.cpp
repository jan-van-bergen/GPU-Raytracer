#include "IO.h"

#include <filesystem>

static std::filesystem::path stringview_to_path(StringView str) {
	return { str.start, str.end };
}

bool IO::file_exists(StringView filename) {
	return std::filesystem::exists(stringview_to_path(filename));
}

bool IO::file_is_newer(StringView filename_a, StringView filename_b) {
	std::filesystem::file_time_type last_write_time_filename_a = std::filesystem::last_write_time(stringview_to_path(filename_a));
	std::filesystem::file_time_type last_write_time_filename_b = std::filesystem::last_write_time(stringview_to_path(filename_b));

	return last_write_time_filename_a < last_write_time_filename_b;
}

String IO::file_read(const String & filename) {
	FILE * file = nullptr;
	fopen_s(&file, filename.data(), "rb");

	if (!file) {
		IO::print("ERROR: Unable to open '{}'!\n"_sv, filename);
		abort();
	}

	size_t file_length = std::filesystem::file_size(stringview_to_path(filename.view()));

	String data(file_length);
	fread_s(data.data(), file_length, 1, file_length, file);
	data.data()[file_length] = '\0';

	fclose(file);
	return data;
}

bool IO::file_write(const String & filename, StringView data) {
	FILE * file = nullptr;
	fopen_s(&file, filename.data(), "wb");

	if (!file) return false;

	fwrite(data.start, 1, data.length(), file);
	fclose(file);

	return true;
}

// Based on: https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file
void IO::export_ppm(const String & filename, int width, int height, const unsigned char * data) {
	FILE * file = nullptr;
	fopen_s(&file, filename.data(), "wb");

	if (!file) {
		IO::print("Failed to export '{}'!\n"_sv, filename);
		return;
	}

	fprintf(file, "P6\n %d\n %d\n %d\n", width, height, 255);
	fwrite(data, sizeof(char), width * height * 3, file);

	fclose(file);
}
