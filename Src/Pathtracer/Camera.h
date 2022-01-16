#pragma once
#include "Math/Vector2.h"
#include "Math/Vector3.h"
#include "Math/Quaternion.h"
#include "Math/Matrix4.h"

struct Camera {
	Vector3    position;
	Quaternion rotation;

	float fov; // Field of View in radians
	float pixel_spread_angle;

	float aperture_radius =  0.1f;
	float focal_distance  = 10.0f;

	float near;
	float far;

	float screen_width;
	float screen_height;

	Vector3 bottom_left_corner, bottom_left_corner_rotated;
	Vector3 x_axis, x_axis_rotated;
	Vector3 y_axis, y_axis_rotated;

	Matrix4 projection;
	Matrix4 view_projection;
	Matrix4 view_projection_prev;

	bool moved;

	Camera(float fov, float near = 0.1f, float far = 300.0f) {
		set_fov(fov);
		this->near = near;
		this->far  = far;
	}

	void resize(int width, int height);

	void set_fov(float fov);

	void update(float delta);

private:
	void recalibrate();
};
