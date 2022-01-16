#pragma once
#include <math.h>

#include "Core/Assertion.h"

struct Vector4 {
	union {
		struct {
			float x, y, z, w;
		};
		float data[4];
	};

	inline Vector4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) { }
	inline Vector4(float f) : x(f), y(f), z(f), w(f) { }
	inline Vector4(const float f[4]) : x(f[0]), y(f[1]), z(f[2]), w(f[3]) { }
	inline Vector4(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) { }

	inline static float length_squared(const Vector4 & vector) {
		return dot(vector, vector);
	}

	inline static float length(const Vector4 & vector) {
		return sqrtf(length_squared(vector));
	}

	inline static Vector4 normalize(const Vector4 & vector) {
		float inv_length = 1.0f / length(vector);
		return Vector4(
			vector.x * inv_length,
			vector.y * inv_length,
			vector.z * inv_length,
			vector.w * inv_length
		);
	}

	inline static float dot(const Vector4 & left, const Vector4 & right) {
		return left.x * right.x + left.y * right.y + left.z * right.z + left.w * right.w;
	}

	inline static Vector4 min(const Vector4 & left, const Vector4 & right) {
		return Vector4(
			left.x < right.x ? left.x : right.x,
			left.y < right.y ? left.y : right.y,
			left.z < right.z ? left.z : right.z,
			left.w < right.w ? left.w : right.w
		);
	}

	inline static Vector4 max(const Vector4 & left, const Vector4 & right) {
		return Vector4(
			left.x > right.x ? left.x : right.x,
			left.y > right.y ? left.y : right.y,
			left.z > right.z ? left.z : right.z,
			left.w > right.w ? left.w : right.w
		);
	}

	inline Vector4 operator+=(const Vector4 & vector) { x += vector.x; y += vector.y; z += vector.z; w += vector.w; return *this; }
	inline Vector4 operator-=(const Vector4 & vector) { x -= vector.x; y -= vector.y; z -= vector.z; w -= vector.w; return *this; }
	inline Vector4 operator*=(const Vector4 & vector) { x *= vector.x; y *= vector.y; z *= vector.z; w *= vector.w; return *this; }
	inline Vector4 operator/=(const Vector4 & vector) { x /= vector.x; y /= vector.y; z /= vector.z; w /= vector.w; return *this; }

	inline Vector4 operator+=(float scalar) {                                   x += scalar;     y += scalar;     z += scalar;     w += scalar;     return *this; }
	inline Vector4 operator-=(float scalar) {                                   x -= scalar;     y -= scalar;     z -= scalar;     w -= scalar;     return *this; }
	inline Vector4 operator*=(float scalar) {                                   x *= scalar;     y *= scalar;     z *= scalar;     w *= scalar;     return *this; }
	inline Vector4 operator/=(float scalar) { float inv_scalar = 1.0f / scalar; x *= inv_scalar; y *= inv_scalar; z *= inv_scalar; w *= inv_scalar; return *this; }

	inline       float & operator[](int index)       { ASSERT(index >= 0 && index < 4); return data[index]; }
	inline const float & operator[](int index) const { ASSERT(index >= 0 && index < 4); return data[index]; }
};

inline Vector4 operator-(const Vector4 & vector) { return Vector4(-vector.x, -vector.y, -vector.z, -vector.w); }

inline Vector4 operator+(const Vector4 & left, const Vector4 & right) { return Vector4(left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w); }
inline Vector4 operator-(const Vector4 & left, const Vector4 & right) { return Vector4(left.x - right.x, left.y - right.y, left.z - right.z, left.w - right.w); }
inline Vector4 operator*(const Vector4 & left, const Vector4 & right) { return Vector4(left.x * right.x, left.y * right.y, left.z * right.z, left.w * right.w); }
inline Vector4 operator/(const Vector4 & left, const Vector4 & right) { return Vector4(left.x / right.x, left.y / right.y, left.z / right.z, left.w / right.w); }

inline Vector4 operator+(const Vector4 & vector, float scalar) {                                   return Vector4(vector.x + scalar,     vector.y + scalar,     vector.z + scalar,     vector.w + scalar); }
inline Vector4 operator-(const Vector4 & vector, float scalar) {                                   return Vector4(vector.x - scalar,     vector.y - scalar,     vector.z - scalar,     vector.w - scalar); }
inline Vector4 operator*(const Vector4 & vector, float scalar) {                                   return Vector4(vector.x * scalar,     vector.y * scalar,     vector.z * scalar,     vector.w * scalar); }
inline Vector4 operator/(const Vector4 & vector, float scalar) { float inv_scalar = 1.0f / scalar; return Vector4(vector.x * inv_scalar, vector.y * inv_scalar, vector.z * inv_scalar, vector.w * inv_scalar); }

inline Vector4 operator+(float scalar, const Vector4 & vector) { return Vector4(scalar + vector.x, scalar + vector.y, scalar + vector.z, scalar + vector.w); }
inline Vector4 operator-(float scalar, const Vector4 & vector) { return Vector4(scalar - vector.x, scalar - vector.y, scalar - vector.z, scalar - vector.w); }
inline Vector4 operator*(float scalar, const Vector4 & vector) { return Vector4(scalar * vector.x, scalar * vector.y, scalar * vector.z, scalar * vector.w); }
inline Vector4 operator/(float scalar, const Vector4 & vector) { return Vector4(scalar / vector.x, scalar / vector.y, scalar / vector.z, scalar / vector.w); }

inline bool operator==(const Vector4 & left, const Vector4 & right) { return left.x == right.x && left.y == right.y && left.z == right.z && left.w == right.w; }
inline bool operator!=(const Vector4 & left, const Vector4 & right) { return left.x != right.x || left.y != right.y || left.z != right.z || left.w != right.w; }
