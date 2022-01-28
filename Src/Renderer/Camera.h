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

	float aperture_radius =  0.0f;
	float focal_distance  = 10.0f;

	float near_plane;
	float far_plane;

	float screen_width;
	float screen_height;

	Vector3 bottom_left_corner, bottom_left_corner_rotated;
	Vector3 x_axis, x_axis_rotated;
	Vector3 y_axis, y_axis_rotated;

	Matrix4 projection;
	Matrix4 view_projection;
	Matrix4 view_projection_prev;

	bool moved;

	Camera(float fov, float near_plane = 0.1f, float far_plane = 300.0f) {
		set_fov(fov);
		this->near_plane = near_plane;
		this->far_plane  = far_plane;
	}

	void resize(int width, int height);

	void set_fov(float fov);

	void update(float delta);

private:
	void recalibrate();
};
