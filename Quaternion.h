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

	// Based on: https://answers.unity.com/questions/467614/what-is-the-source-code-of-quaternionlookrotation.html
	inline static Quaternion look_rotation(Vector3 forward, Vector3 up) {
		forward = Vector3::normalize(forward);
		Vector3 right = Vector3::normalize(Vector3::cross(up, forward));
		up = Vector3::cross(forward, right);

		float m00 = right.x;
		float m01 = right.y;
		float m02 = right.z;
		float m10 = up.x;
		float m11 = up.y;
		float m12 = up.z;
		float m20 = forward.x;
		float m21 = forward.y;
		float m22 = forward.z;

		float trace = (m00 + m11) + m22;

		Quaternion quaternion;

		if (trace > 0.0f) {
			float num = sqrtf(trace + 1.0f);

			quaternion.w = num * 0.5f;
			num = 0.5f / num;
			quaternion.x = (m12 - m21) * num;
			quaternion.y = (m20 - m02) * num;
			quaternion.z = (m01 - m10) * num;

			return quaternion;
		}

		if ((m00 >= m11) && (m00 >= m22)) {
			float num7 = sqrtf(((1.0f + m00) - m11) - m22);
			float num4 = 0.5f / num7;

			quaternion.x = 0.5f * num7;
			quaternion.y = (m01 + m10) * num4;
			quaternion.z = (m02 + m20) * num4;
			quaternion.w = (m12 - m21) * num4;

			return quaternion;
		}

		if (m11 > m22)
		{
			float num6 = sqrtf(((1.0f + m11) - m00) - m22);
			float num3 = 0.5f / num6;

			quaternion.x = (m10 + m01) * num3;
			quaternion.y = 0.5f * num6;
			quaternion.z = (m21 + m12) * num3;
			quaternion.w = (m20 - m02) * num3;

			return quaternion;
		}

		float num5 = sqrtf(((1.0f + m22) - m00) - m11);
		float num2 = 0.5f / num5;

		quaternion.x = (m20 + m02) * num2;
		quaternion.y = (m21 + m12) * num2;
		quaternion.z = 0.5f * num5;
		quaternion.w = (m01 - m10) * num2;

		return quaternion;
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
