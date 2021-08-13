#pragma once
#include <string.h>

struct StringView {
	const char * start;
	const char * end;

	inline char operator[](int index) const { return start[index]; }

	int length() const { return end - start; }

	char * c_str() const {
		int    len = length();
		char * str = new char[len + 1];
		memcpy(str, start, len);
		str[len] = '\0';
		return str;
	}

	StringView substr(size_t offset, size_t len = -1) {
		if (offset + len >= length()) {
			len = length() - offset;
		}
		return { start + offset, start + offset + len };
	}
};

struct StringViewHash {
	size_t operator()(const StringView & str) const {
		static constexpr int p = 31;
		static constexpr int m = 1e9 + 9;

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

	return true;
}

inline bool operator==(const StringView & a, const StringView & b) {
	int length_a = a.length();
	int length_b = b.length();

	for (int i = 0; i < length_a; i++) {
		if (a[i] != b[i]) return false;
	}

	return true;
}
