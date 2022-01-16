#pragma once
#include <string.h>

namespace FNVHash {
	inline size_t hash(const char bytes[], size_t length) {
		// Based on: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
		constexpr size_t FNV_OFFSET_BASIS = 14695981039346656037ui64;
		constexpr size_t FNV_PRIME        = 1099511628211ui64;

		size_t hash = FNV_OFFSET_BASIS;
		for (size_t i = 0; i < sizeof(bytes); i++) {
			hash = hash ^ bytes[i];
			hash = hash * FNV_PRIME;
		}

		return hash;
	}
}

template<typename T>
struct Hash {
	size_t operator()(const T & x) const {
		char bytes[sizeof(T)] = { };
		memcpy(bytes, &x, sizeof(T));

		return FNVHash::hash(bytes, sizeof(T));
	}
};

template<typename T>
struct Cmp {
	size_t operator()(const T & a, const T & b) const {
		return a == b;
	}
};
