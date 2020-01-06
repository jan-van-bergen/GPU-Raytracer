#pragma once
#include "Vector3.h"
#include "Quaternion.h"

#include "Util.h"

struct Camera {
	Vector3    position;	
	Quaternion rotation;

	float fov; // Field of View in radians

	Vector3 top_left_corner, top_left_corner_rotated;
	Vector3 x_axis,          x_axis_rotated;
	Vector3 y_axis,          y_axis_rotated;

	bool moved;

	inline Camera(float fov) : fov(fov) { }

	void resize(int width, int height);

	void update(float delta, const unsigned char * keys);
};
