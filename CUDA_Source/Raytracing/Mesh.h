#pragma once

struct Matrix3x4 {
	float4 row_0;
	float4 row_1;
	float4 row_2;
};

// Transform vector as position (w = 1)
__device__ inline void matrix3x4_transform_position(const Matrix3x4 & matrix, float3 & position) {
	position = make_float3( 
		matrix.row_0.x * position.x + matrix.row_0.y * position.y + matrix.row_0.z * position.z + matrix.row_0.w,
		matrix.row_1.x * position.x + matrix.row_1.y * position.y + matrix.row_1.z * position.z + matrix.row_1.w,
		matrix.row_2.x * position.x + matrix.row_2.y * position.y + matrix.row_2.z * position.z + matrix.row_2.w
	);
}

// Transform vector as direction (w = 0)
__device__ inline void matrix3x4_transform_direction(const Matrix3x4 & matrix, float3 & direction) {
	direction = make_float3(
		matrix.row_0.x * direction.x + matrix.row_0.y * direction.y + matrix.row_0.z * direction.z,
		matrix.row_1.x * direction.x + matrix.row_1.y * direction.y + matrix.row_1.z * direction.z,
		matrix.row_2.x * direction.x + matrix.row_2.y * direction.y + matrix.row_2.z * direction.z
	);
}

__device__ __constant__ int       * mesh_bvh_root_indices;
__device__ __constant__ Matrix3x4 * mesh_transforms;
__device__ __constant__ Matrix3x4 * mesh_transforms_inv;
__device__ __constant__ Matrix3x4 * mesh_transforms_prev;

__device__ inline Matrix3x4 mesh_get_transform(int mesh_id) {
	Matrix3x4 matrix;
	matrix.row_0 = __ldg(&mesh_transforms[mesh_id].row_0);
	matrix.row_1 = __ldg(&mesh_transforms[mesh_id].row_1);
	matrix.row_2 = __ldg(&mesh_transforms[mesh_id].row_2);

	return matrix;
}

__device__ inline Matrix3x4 mesh_get_transform_inv(int mesh_id) {
	Matrix3x4 matrix;
	matrix.row_0 = __ldg(&mesh_transforms_inv[mesh_id].row_0);
	matrix.row_1 = __ldg(&mesh_transforms_inv[mesh_id].row_1);
	matrix.row_2 = __ldg(&mesh_transforms_inv[mesh_id].row_2);

	return matrix;
}

__device__ inline Matrix3x4 mesh_get_transform_prev(int mesh_id) {
	Matrix3x4 matrix;
	matrix.row_0 = __ldg(&mesh_transforms_prev[mesh_id].row_0);
	matrix.row_1 = __ldg(&mesh_transforms_prev[mesh_id].row_1);
	matrix.row_2 = __ldg(&mesh_transforms_prev[mesh_id].row_2);

	return matrix;
}

__device__ inline float mesh_get_scale(int mesh_id) {
	// Scale is stored along the diagonal of the transformation matrix
	// We only care about uniform scale, so the first value is sufficient
	return __ldg(&mesh_transforms[mesh_id].row_0.x);
}
