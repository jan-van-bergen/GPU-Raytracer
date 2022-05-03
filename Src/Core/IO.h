#pragma once
#include <stdio.h>

#include "Format.h"
#include "Allocators/LinearAllocator.h"

namespace IO {
	inline void print(char c) {
		putchar(c);
	}

	inline void print(StringView str) {
		fwrite(str.data(), sizeof(char), str.size(), stdout);
	}

	template<typename ... Args>
	inline void print(StringView fmt, const Args & ... args) {
		LinearAllocator<KILOBYTES(4)> allocator;
		String string = Format(&allocator).format(fmt, args ...);
		print(string.view());
	}

	[[noreturn]]
	inline void exit(int code) {
		__debugbreak();
		::exit(code);
	}

	String get_error_message(errno_t error_code, Allocator * allocator = nullptr);

	bool file_exists(StringView filename);

	bool file_is_newer(StringView filename_a, StringView filename_b);

	String file_read (const String & filename, Allocator * allocator);
	bool   file_write(const String & filename, StringView data);
}
