#pragma once
#include <cassert>
#include <string.h>

#include "Vector3.h"
#include "Quaternion.h"

struct Matrix4 {
	float cells[16];

	inline Matrix4() {
		memset(cells, 0, sizeof(cells));
		cells[0]  = 1.0f;
		cells[5]  = 1.0f;
		cells[10] = 1.0f;
		cells[15] = 1.0f;
	}

	inline float & operator()(int i, int j) { 
		assert(i >= 0 && i < 4); 
		assert(j >= 0 && j < 4); 
		return cells[i + (j << 2)]; 
	}

	inline const float & operator()(int i, int j) const { 
		assert(i >= 0 && i < 4); 
		assert(j >= 0 && j < 4); 
		return cells[i + (j << 2)];
	}

	// Transforms a Vector3 as if the fourth coordinate is 1
	inline static Vector3 transform_position(const Matrix4 & matrix, const Vector3 & position) {
		return Vector3(
			matrix(0, 0) * position.x + matrix(1, 0) * position.y + matrix(2, 0) * position.z + matrix(3, 0),
			matrix(0, 1) * position.x + matrix(1, 1) * position.y + matrix(2, 1) * position.z + matrix(3, 1),
			matrix(0, 2) * position.x + matrix(1, 2) * position.y + matrix(2, 2) * position.z + matrix(3, 2)
		);
	}
	
	// Transforms a Vector3 as if the fourth coordinate is 0
	inline static Vector3 transform_direction(const Matrix4 & matrix, const Vector3 & direction) {
		return Vector3(
			matrix(0, 0) * direction.x + matrix(1, 0) * direction.y + matrix(2, 0) * direction.z,
			matrix(0, 1) * direction.x + matrix(1, 1) * direction.y + matrix(2, 1) * direction.z,
			matrix(0, 2) * direction.x + matrix(1, 2) * direction.y + matrix(2, 2) * direction.z
		);
	}

	inline static Matrix4 create_translation(const Vector3 & translation) {
		Matrix4 result;
		result(3, 0) = translation.x;
		result(3, 1) = translation.y;
		result(3, 2) = translation.z;

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
		result(0, 1) =        2.0f * (xy + wz);
		result(0, 2) =        2.0f * (xz - wy);
		
		result(1, 0) =        2.0f * (xy - wz);
		result(1, 1) = 1.0f - 2.0f * (xx + zz);
		result(1, 2) =        2.0f * (yz + wx);
		
		result(2, 0) =        2.0f * (xz + wy);
		result(2, 1) =        2.0f * (yz - wx);
		result(2, 2) = 1.0f - 2.0f * (xx + yy);
		
		return result;
	}

	inline static Matrix4 perspective(float fov, float aspect, float near, float far) {
		float tan_half_fov = tanf(0.5f * fov);

		Matrix4 result;
		result(0, 0) = 1.0f / (aspect * tan_half_fov);
		result(1, 1) = 1.0f / tan_half_fov;
		result(2, 2) = -(far + near) / (far - near);
		result(2, 3) = -1.0f;
		result(3, 2) = -2.0f * (far * near) / (far - near);
		result(3, 3) = 0.0f;

		return result;
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

	// Based on: http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
	inline static Matrix4 invert(const Matrix4 & matrix) {		
		float inv[16] = {
			 matrix.cells[5] * matrix.cells[10] * matrix.cells[15] - matrix.cells[5]  * matrix.cells[11] * matrix.cells[14] - matrix.cells[9]  * matrix.cells[6] * matrix.cells[15] +
			 matrix.cells[9] * matrix.cells[7]  * matrix.cells[14] + matrix.cells[13] * matrix.cells[6]  * matrix.cells[11] - matrix.cells[13] * matrix.cells[7] * matrix.cells[10],
			-matrix.cells[1] * matrix.cells[10] * matrix.cells[15] + matrix.cells[1]  * matrix.cells[11] * matrix.cells[14] + matrix.cells[9]  * matrix.cells[2] * matrix.cells[15] -
			 matrix.cells[9] * matrix.cells[3]  * matrix.cells[14] - matrix.cells[13] * matrix.cells[2]  * matrix.cells[11] + matrix.cells[13] * matrix.cells[3] * matrix.cells[10],
			 matrix.cells[1] * matrix.cells[6]  * matrix.cells[15] - matrix.cells[1]  * matrix.cells[7]  * matrix.cells[14] - matrix.cells[5]  * matrix.cells[2] * matrix.cells[15] +
			 matrix.cells[5] * matrix.cells[3]  * matrix.cells[14] + matrix.cells[13] * matrix.cells[2]  * matrix.cells[7]  - matrix.cells[13] * matrix.cells[3] * matrix.cells[6],
			-matrix.cells[1] * matrix.cells[6]  * matrix.cells[11] + matrix.cells[1]  * matrix.cells[7]  * matrix.cells[10] + matrix.cells[5]  * matrix.cells[2] * matrix.cells[11] -
			 matrix.cells[5] * matrix.cells[3]  * matrix.cells[10] - matrix.cells[9]  * matrix.cells[2]  * matrix.cells[7]  + matrix.cells[9]  * matrix.cells[3] * matrix.cells[6],
			-matrix.cells[4] * matrix.cells[10] * matrix.cells[15] + matrix.cells[4]  * matrix.cells[11] * matrix.cells[14] + matrix.cells[8]  * matrix.cells[6] * matrix.cells[15] -
			 matrix.cells[8] * matrix.cells[7]  * matrix.cells[14] - matrix.cells[12] * matrix.cells[6]  * matrix.cells[11] + matrix.cells[12] * matrix.cells[7] * matrix.cells[10],
			 matrix.cells[0] * matrix.cells[10] * matrix.cells[15] - matrix.cells[0]  * matrix.cells[11] * matrix.cells[14] - matrix.cells[8]  * matrix.cells[2] * matrix.cells[15] +
			 matrix.cells[8] * matrix.cells[3]  * matrix.cells[14] + matrix.cells[12] * matrix.cells[2]  * matrix.cells[11] - matrix.cells[12] * matrix.cells[3] * matrix.cells[10],
			-matrix.cells[0] * matrix.cells[6]  * matrix.cells[15] + matrix.cells[0]  * matrix.cells[7]  * matrix.cells[14] + matrix.cells[4]  * matrix.cells[2] * matrix.cells[15] -
			 matrix.cells[4] * matrix.cells[3]  * matrix.cells[14] - matrix.cells[12] * matrix.cells[2]  * matrix.cells[7]  + matrix.cells[12] * matrix.cells[3] * matrix.cells[6],
			 matrix.cells[0] * matrix.cells[6]  * matrix.cells[11] - matrix.cells[0]  * matrix.cells[7]  * matrix.cells[10] - matrix.cells[4]  * matrix.cells[2] * matrix.cells[11] +
			 matrix.cells[4] * matrix.cells[3]  * matrix.cells[10] + matrix.cells[8]  * matrix.cells[2]  * matrix.cells[7]  - matrix.cells[8]  * matrix.cells[3] * matrix.cells[6],
			 matrix.cells[4] * matrix.cells[9]  * matrix.cells[15] - matrix.cells[4]  * matrix.cells[11] * matrix.cells[13] - matrix.cells[8]  * matrix.cells[5] * matrix.cells[15] +
			 matrix.cells[8] * matrix.cells[7]  * matrix.cells[13] + matrix.cells[12] * matrix.cells[5]  * matrix.cells[11] - matrix.cells[12] * matrix.cells[7] * matrix.cells[9],
			-matrix.cells[0] * matrix.cells[9]  * matrix.cells[15] + matrix.cells[0]  * matrix.cells[11] * matrix.cells[13] + matrix.cells[8]  * matrix.cells[1] * matrix.cells[15] -
			 matrix.cells[8] * matrix.cells[3]  * matrix.cells[13] - matrix.cells[12] * matrix.cells[1]  * matrix.cells[11] + matrix.cells[12] * matrix.cells[3] * matrix.cells[9],
			 matrix.cells[0] * matrix.cells[5]  * matrix.cells[15] - matrix.cells[0]  * matrix.cells[7]  * matrix.cells[13] - matrix.cells[4]  * matrix.cells[1] * matrix.cells[15] +
			 matrix.cells[4] * matrix.cells[3]  * matrix.cells[13] + matrix.cells[12] * matrix.cells[1]  * matrix.cells[7]  - matrix.cells[12] * matrix.cells[3] * matrix.cells[5],
			-matrix.cells[0] * matrix.cells[5]  * matrix.cells[11] + matrix.cells[0]  * matrix.cells[7]  * matrix.cells[9]  + matrix.cells[4]  * matrix.cells[1] * matrix.cells[11] -
			 matrix.cells[4] * matrix.cells[3]  * matrix.cells[9]  - matrix.cells[8]  * matrix.cells[1]  * matrix.cells[7]  + matrix.cells[8]  * matrix.cells[3] * matrix.cells[5],
			-matrix.cells[4] * matrix.cells[9]  * matrix.cells[14] + matrix.cells[4]  * matrix.cells[10] * matrix.cells[13] + matrix.cells[8]  * matrix.cells[5] * matrix.cells[14] -
			 matrix.cells[8] * matrix.cells[6]  * matrix.cells[13] - matrix.cells[12] * matrix.cells[5]  * matrix.cells[10] + matrix.cells[12] * matrix.cells[6] * matrix.cells[9],
			 matrix.cells[0] * matrix.cells[9]  * matrix.cells[14] - matrix.cells[0]  * matrix.cells[10] * matrix.cells[13] - matrix.cells[8]  * matrix.cells[1] * matrix.cells[14] +
			 matrix.cells[8] * matrix.cells[2]  * matrix.cells[13] + matrix.cells[12] * matrix.cells[1]  * matrix.cells[10] - matrix.cells[12] * matrix.cells[2] * matrix.cells[9],
			-matrix.cells[0] * matrix.cells[5]  * matrix.cells[14] + matrix.cells[0]  * matrix.cells[6]  * matrix.cells[13] + matrix.cells[4]  * matrix.cells[1] * matrix.cells[14] -
			 matrix.cells[4] * matrix.cells[2]  * matrix.cells[13] - matrix.cells[12] * matrix.cells[1]  * matrix.cells[6]  + matrix.cells[12] * matrix.cells[2] * matrix.cells[5],
			 matrix.cells[0] * matrix.cells[5]  * matrix.cells[10] - matrix.cells[0]  * matrix.cells[6]  * matrix.cells[9]  - matrix.cells[4]  * matrix.cells[1] * matrix.cells[10] +
			 matrix.cells[4] * matrix.cells[2]  * matrix.cells[9]  + matrix.cells[8]  * matrix.cells[1]  * matrix.cells[6]  - matrix.cells[8]  * matrix.cells[2] * matrix.cells[5]
		};
		
		Matrix4 result;

		float det = 
			matrix.cells[0] * inv[0] + matrix.cells[1] * inv[4] + 
			matrix.cells[2] * inv[8] + matrix.cells[3] * inv[12];

		if (det != 0.0f) {
			const float inv_det = 1.0f / det;
			for (int i = 0; i < 16; i++) {
				result.cells[i] = inv[i] * inv_det;
			}
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
