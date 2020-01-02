#pragma once
#include <math.h>
#include <cassert>

struct Vector3 {
	union {
		struct {
			float x, y, z;
		};
		float data[3];
	};

	inline Vector3() : x(0.0f), y(0.0f), z(0.0f) { }
	inline Vector3(float f) : x(f), y(f), z(f) {}
	inline Vector3(float x, float y, float z) : x(x), y(y), z(z) { }

	inline static float length_squared(const Vector3 & vector) {
		return dot(vector, vector);
	}

	inline static float length(const Vector3 & vector) {
		return sqrtf(length_squared(vector));
	}

	inline static Vector3 normalize(const Vector3 & vector) {
		float inv_length = 1.0f / length(vector);
		return Vector3(
			vector.x * inv_length, 
			vector.y * inv_length, 
			vector.z * inv_length
		);
	}

	inline static float dot(const Vector3 & left, const Vector3 & right) {
		return left.x * right.x + left.y * right.y + left.z * right.z;
	}

	inline static Vector3 cross(const Vector3 & left, const Vector3 & right) {
		return Vector3(
			left.y * right.z - left.z * right.y,
			left.z * right.x - left.x * right.z,
			left.x * right.y - left.y * right.x
		);
	}

	inline static Vector3 min(const Vector3 & left, const Vector3 & right) {
		return Vector3(
			left.x < right.x ? left.x : right.x,
			left.y < right.y ? left.y : right.y,
			left.z < right.z ? left.z : right.z
		);
	}
	
	inline static Vector3 max(const Vector3 & left, const Vector3 & right) {
		return Vector3(
			left.x > right.x ? left.x : right.x,
			left.y > right.y ? left.y : right.y,
			left.z > right.z ? left.z : right.z
		);
	}

	inline Vector3 operator+=(const Vector3 & vector) { x += vector.x; y += vector.y; z += vector.z; return *this; }
	inline Vector3 operator-=(const Vector3 & vector) { x -= vector.x; y -= vector.y; z -= vector.z; return *this; }
	inline Vector3 operator*=(const Vector3 & vector) { x *= vector.x; y *= vector.y; z *= vector.z; return *this; }
	inline Vector3 operator/=(const Vector3 & vector) { x /= vector.x; y /= vector.y; z /= vector.z; return *this; }

	inline Vector3 operator+=(float scalar) {                                   x += scalar;     y += scalar;     z += scalar;     return *this; }
	inline Vector3 operator-=(float scalar) {                                   x -= scalar;     y -= scalar;     z -= scalar;     return *this; }
	inline Vector3 operator*=(float scalar) {                                   x *= scalar;     y *= scalar;     z *= scalar;     return *this; }
	inline Vector3 operator/=(float scalar) { float inv_scalar = 1.0f / scalar; x *= inv_scalar; y *= inv_scalar; z *= inv_scalar; return *this; }

	inline       float & operator[](int index)       { assert(index >= 0 && index < 3); return data[index]; }
	inline const float & operator[](int index) const { assert(index >= 0 && index < 3); return data[index]; }
};

inline Vector3 operator-(const Vector3 & vector) { return Vector3(-vector.x, -vector.y, -vector.z); }

inline Vector3 operator+(const Vector3 & left, const Vector3 & right) { return Vector3(left.x + right.x, left.y + right.y, left.z + right.z); }
inline Vector3 operator-(const Vector3 & left, const Vector3 & right) { return Vector3(left.x - right.x, left.y - right.y, left.z - right.z); }
inline Vector3 operator*(const Vector3 & left, const Vector3 & right) { return Vector3(left.x * right.x, left.y * right.y, left.z * right.z); }
inline Vector3 operator/(const Vector3 & left, const Vector3 & right) { return Vector3(left.x / right.x, left.y / right.y, left.z / right.z); }

inline Vector3 operator+(const Vector3 & vector, float scalar) {                                   return Vector3(vector.x + scalar,     vector.y + scalar,     vector.z + scalar); }
inline Vector3 operator-(const Vector3 & vector, float scalar) {                                   return Vector3(vector.x - scalar,     vector.y - scalar,     vector.z - scalar); }
inline Vector3 operator*(const Vector3 & vector, float scalar) {                                   return Vector3(vector.x * scalar,     vector.y * scalar,     vector.z * scalar); }
inline Vector3 operator/(const Vector3 & vector, float scalar) { float inv_scalar = 1.0f / scalar; return Vector3(vector.x * inv_scalar, vector.y * inv_scalar, vector.z * inv_scalar); }

inline Vector3 operator+(float scalar, const Vector3 & vector) {                                   return Vector3(vector.x + scalar,     vector.y + scalar,     vector.z + scalar); }
inline Vector3 operator-(float scalar, const Vector3 & vector) {                                   return Vector3(vector.x - scalar,     vector.y - scalar,     vector.z - scalar); }
inline Vector3 operator*(float scalar, const Vector3 & vector) {                                   return Vector3(vector.x * scalar,     vector.y * scalar,     vector.z * scalar); }
inline Vector3 operator/(float scalar, const Vector3 & vector) { float inv_scalar = 1.0f / scalar; return Vector3(vector.x * inv_scalar, vector.y * inv_scalar, vector.z * inv_scalar); }

inline bool operator==(const Vector3 & left, const Vector3 & right) { return left.x == right.x && left.y == right.y && left.z == right.z; }
inline bool operator!=(const Vector3 & left, const Vector3 & right) { return left.x != right.x || left.y != right.y || left.z != right.z; }
