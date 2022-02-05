#pragma once
#include <math.h>

#include "Math.h"
#include "Vector3.h"

struct Quaternion {
	float x, y, z, w;

	inline Quaternion() : x(0.0f), y(0.0f), z(0.0f), w(1.0f) { }
	inline Quaternion(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) { }

	inline static float length(const Quaternion & quaternion) {
		return sqrtf(quaternion.x*quaternion.x + quaternion.y*quaternion.y + quaternion.z*quaternion.z + quaternion.w*quaternion.w);
	}

	inline static Quaternion normalize(const Quaternion & quaternion) {
		float inv_length = 1.0f / length(quaternion);
		return Quaternion(quaternion.x * inv_length, quaternion.y * inv_length, quaternion.z * inv_length, quaternion.w * inv_length);
	}

	inline static Quaternion conjugate(const Quaternion & quaternion) {
		return Quaternion(-quaternion.x, -quaternion.y, -quaternion.z, quaternion.w);
	}

	inline static Quaternion axis_angle(const Vector3 & axis, float angle) {
		float half_angle = 0.5f * angle;
		float sine = sinf(half_angle);

		return Quaternion(
			axis.x * sine,
			axis.y * sine,
			axis.z * sine,
			cosf(half_angle)
		);
	}

	inline static Quaternion look_rotation(const Vector3 & forward, const Vector3 & up) {
		Vector3 f = Vector3::normalize(forward);
		Vector3 r = Vector3::normalize(Vector3::cross(up, f));
		Vector3 u = Vector3::cross(f, r);

		float m00 = r.x; float m01 = r.y; float m02 = r.z;
		float m10 = u.x; float m11 = u.y; float m12 = u.z;
		float m20 = f.x; float m21 = f.y; float m22 = f.z;

		// Based on: https://math.stackexchange.com/a/3183435
		if (m22 < 0.0f) {
			if (m00 > m11) {
				float t = 1.0f + m00 - m11 - m22;
				float s = 0.5f / sqrtf(t);
				return Quaternion(s * t, s * (m01 + m10), s * (m20 + m02), s * (m12 - m21));
			} else {
				float t = 1.0f - m00 + m11 - m22;
				float s = 0.5f / sqrtf(t);
				return Quaternion(s * (m01 + m10), s * t, s * (m12 + m21), s * (m20 - m02));
			}
		} else {
			if (m00 < -m11) {
				float t = 1.0f - m00 - m11 + m22;
				float s = 0.5f / sqrtf(t);
				return Quaternion(s * (m20 + m02), s * (m12 + m21), s * t, s * (m01 - m10));
			} else {
				float t = 1.0f + m00 + m11 + m22;
				float s = 0.5f / sqrtf(t);
				return Quaternion(s * (m12 - m21), s * (m20 - m02), s * (m01 - m10), s * t);
			}
		}
	}

	// Based on: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
	inline static Quaternion from_euler(float yaw, float pitch, float roll) {
		float cos_yaw   = cosf(yaw   * 0.5f);
		float sin_yaw   = sinf(yaw   * 0.5f);
		float cos_pitch = cosf(pitch * 0.5f);
		float sin_pitch = sinf(pitch * 0.5f);
		float cos_roll  = cosf(roll  * 0.5f);
		float sin_roll  = sinf(roll  * 0.5f);

		return {
			sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw,
			cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw,
			cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw,
			cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw
		};
	}

	// Based on: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
	inline static Vector3 to_euler(const Quaternion & q) {
		Vector3 euler;

		// Roll
		float sinr_cosp = 2.0f        * (q.w * q.x + q.y * q.z);
		float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
		euler.z = atan2f(sinr_cosp, cosr_cosp);

		// Pitch
		float sinp = 2.0f * (q.w * q.y - q.z * q.x);
		if (fabsf(sinp) >= 1.0f) {
			euler.y = copysignf(0.5f * PI, sinp); // Use 90 degrees if out of range
		} else {
			euler.y = asinf(sinp);
		}

		// Yaw
		float siny_cosp = 2.0f        * (q.w * q.z + q.x * q.y);
		float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
		euler.x = atan2f(siny_cosp, cosy_cosp);

		// atan2f returns in the range [-pi, pi], remap this to [0, 2pi]
		if (euler.x < 0.0f) euler.x += TWO_PI;
		if (euler.z < 0.0f) euler.z += TWO_PI;

		return euler;
	}

	inline static Quaternion nlerp(const Quaternion & a, const Quaternion & b, float t) {
		float one_minus_t = 1.0f - t;

		return normalize(Quaternion(
			one_minus_t * a.x + t * b.x,
			one_minus_t * a.y + t * b.y,
			one_minus_t * a.z + t * b.z,
			one_minus_t * a.w + t * b.w
		));
	}
};

inline Quaternion operator*(const Quaternion & left, Quaternion & right) {
	return Quaternion(
		left.x * right.w + left.w * right.x + left.y * right.z - left.z * right.y,
		left.y * right.w + left.w * right.y + left.z * right.x - left.x * right.z,
		left.z * right.w + left.w * right.z + left.x * right.y - left.y * right.x,
		left.w * right.w - left.x * right.x - left.y * right.y - left.z * right.z
	);
}

inline Vector3 operator*(const Quaternion & quaternion, const Vector3 & vector) {
	Vector3 q(quaternion.x, quaternion.y, quaternion.z);

	return 2.0f * Vector3::dot(q, vector) * q +
		(quaternion.w * quaternion.w - Vector3::dot(q, q)) * vector +
		2.0f * quaternion.w * Vector3::cross(q, vector);
}
