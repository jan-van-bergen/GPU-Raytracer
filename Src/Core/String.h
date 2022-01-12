#pragma once
#include <string.h>

#include "StringView.h"
#include "Array.h"

struct String {
	static constexpr size_t SSO_SIZE = 16;

	size_t length;
	union {
		char * ptr;
		char   buf[SSO_SIZE];
	};

	constexpr String() : length(0), ptr(nullptr) { }

	constexpr String(size_t length) : length(length), ptr(nullptr) {
		if (length >= SSO_SIZE) {
			ptr = new char[length + 1];
		}
		data()[0] = '\0';
	}

	constexpr String(const char * str) : length(strlen(str)), ptr(nullptr) {
		if (length >= SSO_SIZE) {
			ptr = new char[length + 1];
		}
		memcpy(data(), str, length + 1);
	}

	constexpr String(const char * str, size_t len) : length(len), ptr(nullptr) {
		if (length >= SSO_SIZE) {
			ptr = new char[length + 1];
		}
		memcpy(data(), str, length);
		data()[length] = '\0';
	}

	constexpr String(const StringView & str) : String(str.start, str.length()) { }

	template<size_t N>
	constexpr String(const char (& str)[N]) : length(N - 1), ptr(nullptr) {
		if (length >= SSO_SIZE) {
			ptr = new char[length + 1];
		}
		memcpy(data(), str, length + 1);
	}

	constexpr String(Array<char> && array) : length(0), ptr(nullptr) {
		bool is_null_terminated = array.size() > 0 && array.back() == '\0';
		if (!is_null_terminated) {
			array.push_back('\0');
		}

		length = array.size() - 1;
		if (length >= SSO_SIZE) {
			ptr = array.buffer;
			array.buffer = nullptr;
			return;
		} else {
			memcpy(data(), array.data(), length);
			data()[length] = '\0';
		}
	}

	~String() {
		if (length >= SSO_SIZE && ptr) {
			delete [] ptr;
		}
	}

	constexpr String(const String & str) : length(str.length), ptr(nullptr) {
		if (length >= SSO_SIZE) {
			ptr = new char[length + 1];
		}
		memcpy(data(), str.data(), length + 1);
	}

	constexpr String & operator=(const String & str) {
		if (length >= SSO_SIZE) {
			if (length == str.length) {
				memcpy(data(), str.data(), length + 1);
				return *this;
			}
			delete [] ptr;
		}

		if (str.length >= SSO_SIZE) {
			ptr = new char[str.length + 1];
		}

		length = str.length;
		memcpy(data(), str.data(), length + 1);

		return *this;
	}

	constexpr String(String && str) : length(str.length), ptr(nullptr) {
		if (length < SSO_SIZE) {
			memcpy(buf, str.buf, length + 1);
		} else {
			ptr     = str.ptr;
			str.ptr = nullptr;
		}
	}

	constexpr String & operator=(String && str) {
		if (length >= SSO_SIZE) {
			delete [] ptr;
		}

		length = str.length;
		if (length < SSO_SIZE) {
			memcpy(buf, str.buf, length + 1);
		} else {
			ptr     = str.ptr;
			str.ptr = nullptr;
		}

		return *this;
	}

	constexpr size_t size() const { return length; }

	constexpr       char * data()       { return length < SSO_SIZE ? buf : ptr; }
	constexpr const char * data() const { return length < SSO_SIZE ? buf : ptr; }

	constexpr const char * c_str() const { return data(); }

	StringView view() const { return { data(), data() + size() }; }

	bool is_empty() const { return length == 0; }

	char & operator[](size_t index)       { return data()[index]; }
	char   operator[](size_t index) const { return data()[index]; }

private:
	// Constexpr implementations
	static constexpr size_t strlen(const char * str) {
		const char * cur = str;
		while (*cur != '\0') {
			cur++;
		}
		return cur - str;
	}

	static constexpr void memcpy(char * dst, const char * str, size_t len) {
		for (size_t i = 0; i < len; i++) {
			dst[i] = str[i];
		}
	}
};

inline bool operator==(const String & a, const String & b) {
	if (a.size() != b.size()) return false;

	for (size_t i = 0; i < a.size(); i++) {
		if (a[i] != b[i]) return false;
	}

	return true;
}

inline bool operator!=(const String & a, const String & b) {
	if (a.size() != b.size()) return true;

	for (size_t i = 0; i < a.size(); i++) {
		if (a[i] != b[i]) return true;
	}

	return false;
}

struct StringHash {
	size_t operator()(const String & str) {
		return StringViewHash()(str.view());
	}
};
