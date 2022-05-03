#pragma once
#include <string.h>

#include "Hash.h"

struct StringView {
	const char * start;
	const char * end;

	inline char operator[](size_t index) const { return start[index]; }

	const char * data() const { return start; }
	size_t       size() const { return end - start; }

	bool is_empty() const { return size() == 0; }

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

inline constexpr StringView operator "" _sv(const char * str, size_t length) {
	return StringView::from_c_str(str, length);
}

template<>
struct Hash<StringView> {
	size_t operator()(StringView str) const {
		return FNVHash::hash(str.data(), str.size());
	}
};

template<int N>
inline bool operator==(const StringView & a, const char (&b)[N]) {
	if (a.size() != N - 1) return false;

	for (int i = 0; i < N - 1; i++) {
		if (a[i] != b[i]) return false;
	}

	return true;
}

template<int N>
inline bool operator!=(const StringView & a, const char (&b)[N]) {
	if (a.size() != N - 1) return true;

	for (int i = 0; i < N - 1; i++) {
		if (a[i] == b[i]) return false;
	}

	return true;
}

inline bool operator==(const StringView & a, const char * b) {
	size_t length = a.size();

	for (int i = 0; i < length; i++) {
		if (a[i] != b[i] || b[i] == '\0') return false;
	}

	return b[length] == '\0';
}

inline bool operator==(const StringView & a, const StringView & b) {
	size_t length_b = b.size();
	size_t length_a = a.size();

	if (length_a != length_b) return false;

	for (int i = 0; i < length_a; i++) {
		if (a[i] != b[i]) return false;
	}

	return true;
}
