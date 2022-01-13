#pragma once
#include <string.h>

#include "Hash.h"

struct StringView {
	const char * start;
	const char * end;

	inline char operator[](size_t index) const { return start[index]; }

	size_t length() const { return end - start; }

	bool is_empty() const { return length() == 0; }

	template<int N>
	constexpr static StringView from_c_str(const char (& str)[N]) {
		return { str, str + N - 1 };
	}

	constexpr static StringView from_c_str(const char * str) {
		return from_c_str(str, strlen(str));
	}

	constexpr static StringView from_c_str(const char * str, size_t length) {
		return { str, str + length };
	}
};

inline constexpr StringView operator "" sv(const char * str, size_t length) {
	return StringView::from_c_str(str, length);
}

template<int N>
inline bool operator==(const StringView & a, const char (&b)[N]) {
	if (a.length() != N - 1) return false;

	for (int i = 0; i < N - 1; i++) {
		if (a[i] != b[i]) return false;
	}

	return true;
}

template<int N>
inline bool operator!=(const StringView & a, const char (&b)[N]) {
	if (a.length() != N - 1) return true;

	for (int i = 0; i < N - 1; i++) {
		if (a[i] == b[i]) return false;
	}

	return true;
}

inline bool operator==(const StringView & a, const char * b) {
	int length = a.length();

	for (int i = 0; i < length; i++) {
		if (a[i] != b[i] || b[i] == '\0') return false;
	}

	return b[length] == '\0';
}

inline bool operator==(const StringView & a, const StringView & b) {
	int length_a = a.length();
	int length_b = b.length();

	if (length_a != length_b) return false;

	for (int i = 0; i < length_a; i++) {
		if (a[i] != b[i]) return false;
	}

	return true;
}
