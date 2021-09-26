#include "Camera.h"

#include <cstdio>
#include <cstdint>

#include "Input.h"

void Camera::resize(int width, int height) {
	screen_width  = float(width);
	screen_height = float(height);
	recalibrate();
}

void Camera::set_fov(float fov) {
	this->fov = fov;
	recalibrate();
}

void Camera::recalibrate() {
	float inv_width  = 1.0f / screen_width;
	float inv_height = 1.0f / screen_height;

	float half_width  = 0.5f * screen_width;
	float half_height = 0.5f * screen_height;

	float tan_half_fov = tanf(0.5f * fov);

	// Distance to the viewing plane
	float d = half_height / tan_half_fov;

	// Initialize viewing pyramid vectors
	bottom_left_corner = Vector3(-half_width, -half_height, -d);
	x_axis = Vector3(1.0f, 0.0f, 0.0f);
	y_axis = Vector3(0.0f, 1.0f, 0.0f);

	// Projection matrix (for rasterization)
	projection = Matrix4::perspective(fov, half_width / half_height, near, far);

	// See equation 30 of "Texture Level of Detail Strategies for Real-Time Ray Tracing"
	pixel_spread_angle = atanf(2.0f * tan_half_fov * inv_width);
}

void Camera::update(float delta) {
	// Move Camera around
	moved = false;

	constexpr float MOVEMENT_SPEED = 10.0f;
	constexpr float ROTATION_SPEED =  3.0f;

	float movement_speed = MOVEMENT_SPEED;
	float rotation_speed = ROTATION_SPEED;

	if (Input::is_key_down(SDL_SCANCODE_Z)) { // Slow movement/rotation down
		movement_speed *= 0.1f;
		rotation_speed *= 0.1f;
	}

	Vector3 right   = rotation * Vector3(1.0f, 0.0f,  0.0f);
	Vector3 forward = rotation * Vector3(0.0f, 0.0f, -1.0f);

	if (Input::is_key_down(SDL_SCANCODE_W)) { position += forward * movement_speed * delta; moved = true; }
	if (Input::is_key_down(SDL_SCANCODE_A)) { position -= right   * movement_speed * delta; moved = true; }
	if (Input::is_key_down(SDL_SCANCODE_S)) { position -= forward * movement_speed * delta; moved = true; }
	if (Input::is_key_down(SDL_SCANCODE_D)) { position += right   * movement_speed * delta; moved = true; }

	if (Input::is_key_down(SDL_SCANCODE_LSHIFT)) { position.y -= movement_speed * delta; moved = true; }
	if (Input::is_key_down(SDL_SCANCODE_SPACE))  { position.y += movement_speed * delta; moved = true; }

	if (Input::is_key_down(SDL_SCANCODE_UP))    { rotation = Quaternion::axis_angle(right,                     +rotation_speed * delta) * rotation; moved = true; }
	if (Input::is_key_down(SDL_SCANCODE_DOWN))  { rotation = Quaternion::axis_angle(right,                     -rotation_speed * delta) * rotation; moved = true; }
	if (Input::is_key_down(SDL_SCANCODE_LEFT))  { rotation = Quaternion::axis_angle(Vector3(0.0f, 1.0f, 0.0f), +rotation_speed * delta) * rotation; moved = true; }
	if (Input::is_key_down(SDL_SCANCODE_RIGHT)) { rotation = Quaternion::axis_angle(Vector3(0.0f, 1.0f, 0.0f), -rotation_speed * delta) * rotation; moved = true; }

	// For debugging purposes
	if (Input::is_key_pressed(SDL_SCANCODE_F)) {
		printf("camera.position = Vector3(%ff, %ff, %ff);\n",         position.x, position.y, position.z);
		printf("camera.rotation = Quaternion(%ff, %ff, %ff, %ff);\n", rotation.x, rotation.y, rotation.z, rotation.w);
	}
	if (Input::is_key_pressed(SDL_SCANCODE_G)) {
		rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f); // Reset
	}

	// Transform view pyramid according to rotation
	bottom_left_corner_rotated = rotation * bottom_left_corner;
	x_axis_rotated             = rotation * x_axis;
	y_axis_rotated             = rotation * y_axis;

	view_projection_prev = view_projection;

	view_projection =
		projection *
		Matrix4::create_rotation(Quaternion::conjugate(rotation)) *
		Matrix4::create_translation(-position);
}
