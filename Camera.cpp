#include "Camera.h"

#include <cstdio>

#include <SDL2/SDL.h>

#include "Common.h"

void Camera::resize(int width, int height) {
	float half_width  = 0.5f * width;
	float half_height = 0.5f * height;

	// Distance to the viewing plane
	float d = half_height / tanf(0.5f * fov);

	// Initialize viewing pyramid vectors
	bottom_left_corner = Vector3(-half_width, -half_height, -d);
	x_axis             = Vector3(1.0f, 0.0f, 0.0f);
	y_axis             = Vector3(0.0f, 1.0f, 0.0f);

	projection = Matrix4::perspective(fov, half_width / half_height, 0.1f, 250.0f);
}

void Camera::update(float delta, const unsigned char * keys) {
	moved = false;

	// Move Camera around
	const float MOVEMENT_SPEED = 10.0f;
	const float ROTATION_SPEED =  3.0f;

	Vector3 right   = rotation * Vector3(1.0f, 0.0f,  0.0f);
	Vector3 forward = rotation * Vector3(0.0f, 0.0f, -1.0f);

	if (keys[SDL_SCANCODE_W]) { position += forward * MOVEMENT_SPEED * delta; moved = true; }
	if (keys[SDL_SCANCODE_A]) { position -= right   * MOVEMENT_SPEED * delta; moved = true; }
	if (keys[SDL_SCANCODE_S]) { position -= forward * MOVEMENT_SPEED * delta; moved = true; }
	if (keys[SDL_SCANCODE_D]) { position += right   * MOVEMENT_SPEED * delta; moved = true; }

	if (keys[SDL_SCANCODE_LSHIFT]) { position.y -= MOVEMENT_SPEED * delta; moved = true; }
	if (keys[SDL_SCANCODE_SPACE])  { position.y += MOVEMENT_SPEED * delta; moved = true; }

	if (keys[SDL_SCANCODE_UP])    { rotation = Quaternion::axis_angle(right,                     +ROTATION_SPEED * delta) * rotation; moved = true; }
	if (keys[SDL_SCANCODE_DOWN])  { rotation = Quaternion::axis_angle(right,                     -ROTATION_SPEED * delta) * rotation; moved = true; }
	if (keys[SDL_SCANCODE_LEFT])  { rotation = Quaternion::axis_angle(Vector3(0.0f, 1.0f, 0.0f), +ROTATION_SPEED * delta) * rotation; moved = true; }
	if (keys[SDL_SCANCODE_RIGHT]) { rotation = Quaternion::axis_angle(Vector3(0.0f, 1.0f, 0.0f), -ROTATION_SPEED * delta) * rotation; moved = true; }

	// For debugging purposes
	if (keys[SDL_SCANCODE_F]) {
		printf("camera.position = Vector3(%ff, %ff, %ff);\n",         position.x, position.y, position.z);
		printf("camera.rotation = Quaternion(%ff, %ff, %ff, %ff);\n", rotation.x, rotation.y, rotation.z, rotation.w);
	}

	///////////////////////////////////////////////////////////
	static unsigned char last = 0;
	if (keys[SDL_SCANCODE_Q] && keys[SDL_SCANCODE_Q] != last) {
		rasterize = !rasterize;

		printf("Rasterization: %s                                                       \n", rasterize ? "On" : "Off");

		moved = true;
	}
	last = keys[SDL_SCANCODE_Q];
	///////////////////////////////////////////////////////////

	// Transform view pyramid according to rotation
	bottom_left_corner_rotated = rotation * bottom_left_corner;
	x_axis_rotated             = rotation * x_axis;
	y_axis_rotated             = rotation * y_axis;

	view_projection_prev = view_projection;

	jitter = Vector2(
		(rng(gen) * 2.0f - 1.0f) * (1.0f / float(SCREEN_WIDTH)), 
		(rng(gen) * 2.0f - 1.0f) * (1.0f / float(SCREEN_HEIGHT))
	);
		
	jitter = Vector2(0.0f);

	// The view matrix V is the inverse of the World M
	// M^-1 = (RT)^-1 = T^-1 * R^-1
	view_projection = 
		Matrix4::create_translation(-position) * 
		Matrix4::create_rotation(Quaternion::conjugate(rotation)) * 
		projection *
		// Apply screen space jitter in NDC
		Matrix4::create_translation(Vector3(jitter.x, jitter.y, 0.0f));
}
