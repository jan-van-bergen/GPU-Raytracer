#pragma once
#include <math.h>

struct Vector2 {
	float x, y;

	inline Vector2() : x(0.0f), y(0.0f) { }
	inline Vector2(float f) : x(f), y(f) {}
	inline Vector2(float x, float y) : x(x), y(y) { }

	inline static float length_squared(const Vector2 & vector) {
		return dot(vector, vector);
	}

	inline static float length(const Vector2 & vector) {
		return sqrtf(length_squared(vector));
	}

	inline static Vector2 normalize(const Vector2 & vector) {
		float inv_length = 1.0f / length(vector);
		return Vector2(vector.x * inv_length, vector.y * inv_length);
	}

	inline static float dot(const Vector2 & left, const Vector2 & right) {
		return left.x * right.x + left.y * right.y;
	}

	inline Vector2 operator+=(const Vector2 & vector) { x += vector.x; y += vector.y; return *this; }
	inline Vector2 operator-=(const Vector2 & vector) { x -= vector.x; y -= vector.y; return *this; }
	inline Vector2 operator*=(const Vector2 & vector) { x *= vector.x; y *= vector.y; return *this; }
	inline Vector2 operator/=(const Vector2 & vector) { x /= vector.x; y /= vector.y; return *this; }

	inline Vector2 operator+=(float scalar) {                                   x += scalar;     y += scalar;     return *this; }
	inline Vector2 operator-=(float scalar) {                                   x -= scalar;     y -= scalar;     return *this; }
	inline Vector2 operator*=(float scalar) {                                   x *= scalar;     y *= scalar;     return *this; }
	inline Vector2 operator/=(float scalar) { float inv_scalar = 1.0f / scalar; x *= inv_scalar; y *= inv_scalar; return *this; }
};

inline Vector2 operator-(const Vector2 & vector) { return Vector2(-vector.x, -vector.y); }

inline Vector2 operator+(const Vector2 & left, const Vector2 & right) { return Vector2(left.x + right.x, left.y + right.y); }
inline Vector2 operator-(const Vector2 & left, const Vector2 & right) { return Vector2(left.x - right.x, left.y - right.y); }
inline Vector2 operator*(const Vector2 & left, const Vector2 & right) { return Vector2(left.x * right.x, left.y * right.y); }
inline Vector2 operator/(const Vector2 & left, const Vector2 & right) { return Vector2(left.x / right.x, left.y / right.y); }

inline Vector2 operator+(const Vector2 & vector, float scalar) {                                   return Vector2(vector.x + scalar,     vector.y + scalar); }
inline Vector2 operator-(const Vector2 & vector, float scalar) {                                   return Vector2(vector.x - scalar,     vector.y - scalar); }
inline Vector2 operator*(const Vector2 & vector, float scalar) {                                   return Vector2(vector.x * scalar,     vector.y * scalar); }
inline Vector2 operator/(const Vector2 & vector, float scalar) { float inv_scalar = 1.0f / scalar; return Vector2(vector.x * inv_scalar, vector.y * inv_scalar);  }

inline Vector2 operator+(float scalar, const Vector2 & vector) {                                   return Vector2(vector.x + scalar,     vector.y + scalar); }
inline Vector2 operator-(float scalar, const Vector2 & vector) {                                   return Vector2(vector.x - scalar,     vector.y - scalar); }
inline Vector2 operator*(float scalar, const Vector2 & vector) {                                   return Vector2(vector.x * scalar,     vector.y * scalar); }
inline Vector2 operator/(float scalar, const Vector2 & vector) { float inv_scalar = 1.0f / scalar; return Vector2(vector.x * inv_scalar, vector.y * inv_scalar);  }

inline bool operator==(const Vector2 & left, const Vector2 & right) { return left.x == right.x && left.y == right.y; }
inline bool operator!=(const Vector2 & left, const Vector2 & right) { return left.x != right.x || left.y != right.y; }
