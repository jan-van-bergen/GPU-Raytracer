#pragma once
#include <random>

#include "Vector2.h"
#include "Vector3.h"
#include "Quaternion.h"
#include "Matrix4.h"

#include "Util.h"

struct Camera {
	Vector3    position;	
	Quaternion rotation;

	float fov; // Field of View in radians

	Vector3 bottom_left_corner, bottom_left_corner_rotated;
	Vector3 x_axis,             x_axis_rotated;
	Vector3 y_axis,             y_axis_rotated;

	Matrix4 projection;
	Matrix4 view_projection;
	Matrix4 view_projection_prev;

	bool moved;

	bool rasterize = true;
	
	std::mt19937                          gen = std::mt19937(std::random_device()());
	std::uniform_real_distribution<float> rng = std::uniform_real_distribution<float>(0.0f, 1.0f);

	inline void init(float fov) {
		this->fov = fov;
	}

	void resize(int width, int height);

	void update(float delta, const unsigned char * keys);
};
