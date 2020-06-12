#pragma once
#include "Vector2.h"
#include "Vector3.h"
#include "Quaternion.h"
#include "Matrix4.h"

struct Camera {
	Vector3    position;	
	Quaternion rotation;

	float fov; // Field of View in radians

	float near;
	float far;

	Vector3 bottom_left_corner, bottom_left_corner_rotated;
	Vector3 x_axis,             x_axis_rotated;
	Vector3 y_axis,             y_axis_rotated;

	Matrix4 projection;
	Matrix4 view_projection;
	Matrix4 view_projection_prev;

	bool moved;

	Vector2 jitter;
	int     jitter_index = 0;

	inline void init(float fov, float near = 0.1f, float far = 300.0f) {
		this->fov = fov;
		this->near = near;
		this->far  = far;
	}

	void resize(int width, int height);

	void update(float delta, const unsigned char * keys, bool apply_jitter);
};
