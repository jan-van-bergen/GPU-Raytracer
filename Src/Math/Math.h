#pragma once
#include <float.h>

#include "Math/Vector3.h"

#include "../CUDA_Source/Common.h"

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
	inline constexpr T wrap(T value, T min, T max) {
		while (value < min) value += max - min;
		while (value > max) value -= max - min;

		return value;
	}

	// Based on: https://stackoverflow.com/questions/4915462/how-should-i-do-floating-point-comparison
	inline bool approx_equal(float a, float b, float epsilon = 0.0001f) {
		float abs_a = fabsf(a);
		float abs_b = fabsf(b);
		float diff  = fabsf(a - b);
		if (a == b) {
			// Shortcut, handles infinities
			return true;
		} else if (a == 0.0f || b == 0.0f || diff < FLT_MIN) {
			// a or b is zero or both are extremely close to it
			// relative error is less meaningful here
			return diff < (epsilon * FLT_MIN);
		} else {
			// Use relative error
			return diff / (abs_a + abs_b) < epsilon;
		}
	}

	template<typename T>
	inline constexpr T divide_round_up(T numerator, T denominator) {
		return (numerator + denominator - 1) / denominator;
	}

	template<typename T>
	inline constexpr T round_up(T x, T n) {
		T remainder = x % n;
		if (remainder == 0) {
			return x;
		} else {
			return x + (n - remainder);
		}
	}

	template<typename T> inline constexpr T square(T x) { return x * x; }

	template<typename T> inline constexpr T min(T a, T b) { return a < b ? a : b;}
	template<typename T> inline constexpr T max(T a, T b) { return a > b ? a : b;}

	template<typename T>
	inline constexpr T lerp(const T & a, const T & b, float t) {
		return (1.0f - t) * a + t * b;
	}

	template<typename T>
	inline T inv_lerp(const T & value, const T & min, const T & max) {
		return (value - min) / (max - min);
	}

	inline Vector3 orthogonal(const Vector3 & v) {
		float s = copysignf(1.0f, v.z);
		float a = -1.0f / (s + v.z);
		float b = v.x * v.y * a;

		return Vector3(1.0f + s * v.x * v.x * a, s * b, -s * v.x);
	};

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

	inline constexpr float luminance(float r, float g, float b) {
		return 0.299f * r + 0.587f * g + 0.114f * b;
	}

	inline constexpr float rad_to_deg(float rad) { return rad * ONE_OVER_PI * 180.0f; }
	inline constexpr float deg_to_rad(float deg) { return deg / 180.0f * PI; }

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

	template<typename Function>
	inline constexpr float invert_monotonically_increasing_function(float y, Function function, float x_min = 0.0f, float x_max = 1.0f) {
		float x = { };

		// Binary search
		for (int i = 0; i < 100; i++) {
			x = 0.5f * (x_min + x_max);
			float y_approx = function(x);
			if (Math::approx_equal(y_approx, y)) break;

			if (y_approx > y) { // Assumes function is monotonically increasing
				x_max = x;
			} else {
				x_min = x;
			}
		}

		return x;
	};

}
