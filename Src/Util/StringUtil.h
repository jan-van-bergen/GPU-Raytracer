#pragma once
#include "String.h"
#include "StringView.h"

namespace Util {
	StringView get_directory(StringView filename);
	StringView remove_directory(StringView filename);

	StringView get_file_extension(StringView filename);

	StringView substr(StringView str, size_t offset, size_t len = -1);

	String combine_stringviews(StringView path, StringView filename);

	const char * find_last_after(StringView haystack, StringView needles);

	const char * strstr(StringView haystack, StringView needle);
}
