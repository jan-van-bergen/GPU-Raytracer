#pragma once
#include <stdio.h>

#include "Format.h"

namespace IO {
	inline void print(char c) {
		putchar(c);
	}

	inline void print(StringView str) {
		fwrite(str.start, sizeof(char), str.length(), stdout);
	}

	template<typename ... Args>
	inline void print(StringView fmt, const Args & ... args) {
		String string = Format().format(fmt, args ...);
		print(string.view());
	}

	bool file_exists(StringView filename);

	bool file_is_newer(StringView filename_a, StringView filename_b);

	String file_read (const String & filename);
	bool   file_write(const String & filename, StringView data);

	void export_ppm(const String & filename, int width, int height, const unsigned char * data);
}
