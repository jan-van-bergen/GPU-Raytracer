#pragma once
#include <cstring>

#include "Vector3.h"
#include "Vector4.h"
#include "Quaternion.h"

#include "Util/Util.h"

struct alignas(16) Matrix4 {
	float cells[16];

	inline Matrix4() {
		memset(cells, 0, sizeof(cells));
		cells[0]  = 1.0f;
		cells[5]  = 1.0f;
		cells[10] = 1.0f;
		cells[15] = 1.0f;
	}

	inline FORCE_INLINE float & operator()(int row, int col) {
		ASSERT(row >= 0 && row < 4);
		ASSERT(col >= 0 && col < 4);
		return cells[col + (row << 2)];
	}

	inline FORCE_INLINE const float & operator()(int row, int col) const {
		ASSERT(row >= 0 && row < 4);
		ASSERT(col >= 0 && col < 4);
		return cells[col + (row << 2)];
	}

	inline static Matrix4 create_translation(const Vector3 & translation) {
		Matrix4 result;
		result(0, 3) = translation.x;
		result(1, 3) = translation.y;
		result(2, 3) = translation.z;

		return result;
	}

	inline static Matrix4 create_rotation(const Quaternion & rotation) {
		float xx = rotation.x * rotation.x;
		float yy = rotation.y * rotation.y;
		float zz = rotation.z * rotation.z;
		float xz = rotation.x * rotation.z;
		float xy = rotation.x * rotation.y;
		float yz = rotation.y * rotation.z;
		float wx = rotation.w * rotation.x;
		float wy = rotation.w * rotation.y;
		float wz = rotation.w * rotation.z;

		Matrix4 result;
		result(0, 0) = 1.0f - 2.0f * (yy + zz);
		result(1, 0) =        2.0f * (xy + wz);
		result(2, 0) =        2.0f * (xz - wy);

		result(0, 1) =        2.0f * (xy - wz);
		result(1, 1) = 1.0f - 2.0f * (xx + zz);
		result(2, 1) =        2.0f * (yz + wx);

		result(0, 2) =        2.0f * (xz + wy);
		result(1, 2) =        2.0f * (yz - wx);
		result(2, 2) = 1.0f - 2.0f * (xx + yy);

		return result;
	}

	inline static Matrix4 create_scale(float scale) {
		Matrix4 result;
		result(0, 0) = scale;
		result(1, 1) = scale;
		result(2, 2) = scale;

		return result;
	}

	inline static Matrix4 create_scale(float x, float y, float z) {
		Matrix4 result;
		result(0, 0) = x;
		result(1, 1) = y;
		result(2, 2) = z;

		return result;
	}

	inline static Matrix4 perspective(float fov, float aspect, float near_plane, float far_plane) {
		float tan_half_fov = tanf(0.5f * fov);

		Matrix4 result;
		result(0, 0) = 1.0f / tan_half_fov;
		result(1, 1) = 1.0f / (aspect * tan_half_fov);
		result(2, 2) = -(far_plane + near_plane) / (far_plane - near_plane);
		result(3, 2) = -1.0f;
		result(2, 3) = -2.0f * (far_plane * near_plane) / (far_plane - near_plane);
		result(3, 3) = 0.0f;

		return result;
	}

	inline static Matrix4 perspective_infinite(float fov, float aspect, float near_plane) {
		float tan_half_fov = tanf(0.5f * fov);

		Matrix4 result;
		result(0, 0) =  1.0f / (aspect * tan_half_fov);
		result(1, 1) =  1.0f / tan_half_fov;
		result(2, 2) = -1.0f;
		result(3, 2) = -1.0f;
		result(2, 3) = -2.0f * near_plane;
		result(3, 3) =  0.0f;

		return result;
	}

	// Transforms a Vector3 as if the fourth coordinate is 1
	inline static Vector3 transform_position(const Matrix4 & matrix, const Vector3 & position) {
		return Vector3(
			matrix(0, 0) * position.x + matrix(0, 1) * position.y + matrix(0, 2) * position.z + matrix(0, 3),
			matrix(1, 0) * position.x + matrix(1, 1) * position.y + matrix(1, 2) * position.z + matrix(1, 3),
			matrix(2, 0) * position.x + matrix(2, 1) * position.y + matrix(2, 2) * position.z + matrix(2, 3)
		);
	}

	// Transforms a Vector3 as if the fourth coordinate is 0
	inline static Vector3 transform_direction(const Matrix4 & matrix, const Vector3 & direction) {
		return Vector3(
			matrix(0, 0) * direction.x + matrix(0, 1) * direction.y + matrix(0, 2) * direction.z,
			matrix(1, 0) * direction.x + matrix(1, 1) * direction.y + matrix(1, 2) * direction.z,
			matrix(2, 0) * direction.x + matrix(2, 1) * direction.y + matrix(2, 2) * direction.z
		);
	}

	inline static Vector4 transform(const Matrix4 & matrix, const Vector4 & vector) {
		return Vector4(
			matrix(0, 0) * vector.x + matrix(0, 1) * vector.y + matrix(0, 2) * vector.z + matrix(0, 3) * vector.w,
			matrix(1, 0) * vector.x + matrix(1, 1) * vector.y + matrix(1, 2) * vector.z + matrix(1, 3) * vector.w,
			matrix(2, 0) * vector.x + matrix(2, 1) * vector.y + matrix(2, 2) * vector.z + matrix(2, 3) * vector.w,
			matrix(3, 0) * vector.x + matrix(3, 1) * vector.y + matrix(3, 2) * vector.z + matrix(3, 3) * vector.w
		);
	}

	inline static Matrix4 transpose(const Matrix4 & matrix) {
		Matrix4 result;

		for (int j = 0; j < 4; j++) {
			for (int i = 0; i < 4; i++) {
				result(i, j) = matrix(j, i);
			}
		}

		return result;
	}

	// Based on: https://github.com/graphitemaster/normals_revisited
	inline static float minor(const Matrix4 & matrix, int r0, int r1, int r2, int c0, int c1, int c2) {
	  return
			matrix(r0, c0) * (matrix(r1, c1) * matrix(r2, c2) - matrix(r2, c1) * matrix(r1, c2)) -
			matrix(r0, c1) * (matrix(r1, c0) * matrix(r2, c2) - matrix(r2, c0) * matrix(r1, c2)) +
			matrix(r0, c2) * (matrix(r1, c0) * matrix(r2, c1) - matrix(r2, c0) * matrix(r1, c1));
	}

	// Based on: https://github.com/graphitemaster/normals_revisited
	inline static Matrix4 cofactor(const Matrix4 & matrix) {
		Matrix4 result;
		result(0, 0) =  minor(matrix, 1, 2, 3, 1, 2, 3);
		result(0, 1) = -minor(matrix, 1, 2, 3, 0, 2, 3);
		result(0, 2) =  minor(matrix, 1, 2, 3, 0, 1, 3);
		result(0, 3) = -minor(matrix, 1, 2, 3, 0, 1, 2);
		result(1, 0) = -minor(matrix, 0, 2, 3, 1, 2, 3);
		result(1, 1) =  minor(matrix, 0, 2, 3, 0, 2, 3);
		result(1, 2) = -minor(matrix, 0, 2, 3, 0, 1, 3);
		result(1, 3) =  minor(matrix, 0, 2, 3, 0, 1, 2);
		result(2, 0) =  minor(matrix, 0, 1, 3, 1, 2, 3);
		result(2, 1) = -minor(matrix, 0, 1, 3, 0, 2, 3);
		result(2, 2) =  minor(matrix, 0, 1, 3, 0, 1, 3);
		result(2, 3) = -minor(matrix, 0, 1, 3, 0, 1, 2);
		result(3, 0) = -minor(matrix, 0, 1, 2, 1, 2, 3);
		result(3, 1) =  minor(matrix, 0, 1, 2, 0, 2, 3);
		result(3, 2) = -minor(matrix, 0, 1, 2, 0, 1, 3);
		result(3, 3) =  minor(matrix, 0, 1, 2, 0, 1, 2);
		return result;
	}

	inline static void decompose(const Matrix4 & matrix, Vector3 * position, Quaternion * rotation, float * scale, const Vector3 & forward = Vector3(0.0f, 0.0f, -1.0f)) {
		if (position) *position = Vector3(matrix(0, 3), matrix(1, 3), matrix(2, 3));
		if (rotation) *rotation = Quaternion::look_rotation(Matrix4::transform_direction(matrix, forward), Vector3(0.0f, 1.0f, 0.0f));
		if (scale) {
			float scale_x = Vector3::length(Vector3(matrix(0, 0), matrix(0, 1), matrix(0, 2)));
			float scale_y = Vector3::length(Vector3(matrix(1, 0), matrix(1, 1), matrix(1, 2)));
			float scale_z = Vector3::length(Vector3(matrix(2, 0), matrix(2, 1), matrix(2, 2)));

			*scale = cbrtf(scale_x * scale_y * scale_z);
		}
	}

	// Component-wise absolute value
	inline static Matrix4 abs(const Matrix4 & matrix) {
		Matrix4 result;

		for (int i = 0; i < 16; i++) {
			result.cells[i] = fabsf(matrix.cells[i]);
		}

		return result;
	}
};

inline Matrix4 operator*(const Matrix4 & left, const Matrix4 & right) {
	Matrix4 result;

	for (int j = 0; j < 4; j++) {
		for (int i = 0; i < 4; i++) {
			result(i, j) =
				left(i, 0) * right(0, j) +
				left(i, 1) * right(1, j) +
				left(i, 2) * right(2, j) +
				left(i, 3) * right(3, j);
		}
	}

	return result;
}
