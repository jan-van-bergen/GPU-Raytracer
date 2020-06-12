#pragma once
#include "Vector3.h"

// Various math util functions
namespace Math {
	// Clamps the value between a smallest and largest allowed value
	template<typename T>
	inline T clamp(T value, T min, T max) {
		if (value < min) return min;
		if (value > max) return max;

		return value;
	}

	// Checks if n is a power of two
	inline constexpr bool is_power_of_two(int n) {
		if (n == 0) return false;

		return (n & (n - 1)) == 0;
	}
	
	// Computes positive modulo of given value
	inline unsigned mod(int value, int modulus) {
		int result = value % modulus;
		if (result < 0) {
			result += modulus;
		}

		return result;
	}

	// Calculates N-th power by repeated squaring. This only works when N is a power of 2
	template<int N> inline float pow2(float value);

	template<>      inline float pow2<0>(float value) { return 1.0f; }
	template<>      inline float pow2<1>(float value) { return value; }
	template<int N> inline float pow2   (float value) { static_assert(is_power_of_two(N)); float sqrt = pow2<N / 2>(value); return sqrt * sqrt; }

	template<int N> inline double pow2(double value);

	template<>      inline double pow2<0>(double value) { return 1.0; }
	template<>      inline double pow2<1>(double value) { return value; }
	template<int N> inline double pow2   (double value) { static_assert(is_power_of_two(N)); double sqrt = pow2<N / 2>(value); return sqrt * sqrt; }
}
