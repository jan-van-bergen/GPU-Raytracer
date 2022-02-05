#pragma once
#include <string.h>

#define FORCEINLINE __forceinline

namespace Util {
	template<typename T>
	constexpr void swap(T & a, T & b) {
		T tmp = std::move(a);
		a     = std::move(b);
		b     = std::move(tmp);
	}

	template<typename To, typename From>
	constexpr To bit_cast(const From & value) {
		static_assert(sizeof(From) == sizeof(To));

		To result = { };
		memcpy(&result, &value, sizeof(From));
		return result;
	}

	template<typename T>
	constexpr void reverse(T * array, size_t length) {
		for (size_t i = 0; i < length / 2; i++) {
			Util::swap(array[i], array[length - i - 1]);
		}
	}

	template<typename T, int N>
	constexpr int array_count(const T (& array)[N]) {
		return N;
	}
}
