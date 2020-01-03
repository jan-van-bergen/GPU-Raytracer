#include "Util.h"

#include <cstring>

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
