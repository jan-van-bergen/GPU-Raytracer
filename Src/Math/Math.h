#pragma once
#include "Math/Vector3.h"

// Various math util functions
namespace Math {
	// Clamps the value between a smallest and largest allowed value
	template<typename T>
	inline constexpr T clamp(T value, T min, T max) {
		if (value < min) return min;
		if (value > max) return max;

		return value;
	}

	template<typename T>
	inline constexpr T divide_round_up(T numerator, T denominator) {
		return (numerator + denominator - 1) / denominator;
	}

	template<typename T> inline constexpr T min(T a, T b) { return a < b ? a : b;}
	template<typename T> inline constexpr T max(T a, T b) { return a > b ? a : b;}

	inline constexpr float linear_to_gamma(float x) {
		if (x <= 0.0f) {
			return 0.0f;
		} else if (x >= 1.0f) {
			return 1.0f;
		} else if (x < 0.0031308f) {
			return x * 12.92f;
		} else {
			return powf(x, 1.0f / 2.4f) * 1.055f - 0.055f;
		}
	}

	inline constexpr float gamma_to_linear(float x) {
		if (x <= 0.0f) {
			return 0.0f;
		} else if (x >= 1.0f) {
			return 1.0f;
		} else if (x < 0.04045f) {
			return x / 12.92f;
		} else {
			return powf((x + 0.055f) / 1.055f, 2.4f);
		}
	}

	// Checks if n is a power of two
	inline constexpr bool is_power_of_two(int n) {
		if (n == 0) return false;

		return (n & (n - 1)) == 0;
	}
	
	// Computes positive modulo of given value
	inline constexpr unsigned mod(int value, int modulus) {
		int result = value % modulus;
		if (result < 0) {
			result += modulus;
		}

		return result;
	}
	
	// Based on Jonathan Blow's GD mag code
	inline float sincf(float x) {
		if (fabsf(x) < 0.0001f) {
			return 1.0f + x*x*(-1.0f/6.0f + x*x*1.0f/120.0f);
		} else {
			return sinf(x) / x;
		}
	}

	// Based on Jonathan Blow's GD mag code
	inline constexpr float bessel_0(float x) {
		constexpr float EPSILON_RATIO = 1e-6f;

		float xh  = 0.5f * x;
		float sum = 1.0f;
		float pow = 1.0f;
		float ds  = 1.0;
		float k   = 0.0f;

		while (ds > sum * EPSILON_RATIO) {
			k += 1.0f;
			pow = pow * (xh / k);
			ds  = pow * pow;
			sum = sum + ds;
		}

		return sum;
	}
}
