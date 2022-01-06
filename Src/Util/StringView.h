#pragma once
#include <string.h>

#define FMT_STRINGVIEW(str_view) unsigned(str_view.length()), str_view.start

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

	static StringView from_c_str(const char * str) {
		return { str, str + strlen(str) };
	}
};

struct StringViewHash {
	// Based on: https://www.geeksforgeeks.org/string-hashing-using-polynomial-rolling-hash-function/
	size_t operator()(const StringView & str) const {
		static constexpr int p = 31;
		static constexpr int m = 1'000'000'009;

		size_t hash = 0;
		size_t p_pow = 1;
		for (int i = 0; i < str.length(); i++) {
			hash = (hash + (str[i] - 'a' + 1) * p_pow) % m;
			p_pow = (p_pow * p) % m;
		}

		return hash;
	}
};

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
