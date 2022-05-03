#include "IO.h"

#include <filesystem>

String IO::get_error_message(errno_t error_code, Allocator * allocator) {
	char error_message[512];
	strerror_s(error_message, error_code);
	return String(static_cast<const char *>(error_message), allocator);
}

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

String IO::file_read(const String & filename, Allocator * allocator) {
	FILE * file = nullptr;
	errno_t err = fopen_s(&file, filename.data(), "rb");

	if (!file) {
		IO::print("ERROR: Unable to open '{}'! ({})\n"_sv, filename, get_error_message(err));
		IO::exit(1);
	}

	size_t file_length = std::filesystem::file_size(stringview_to_path(filename.view()));

	String data(file_length, allocator);
	fread_s(data.data(), file_length, 1, file_length, file);
	data.data()[file_length] = '\0';

	fclose(file);
	return data;
}

bool IO::file_write(const String & filename, StringView data) {
	FILE * file = nullptr;
	fopen_s(&file, filename.data(), "wb");

	if (!file) return false;

	fwrite(data.data(), 1, data.size(), file);
	fclose(file);

	return true;
}
