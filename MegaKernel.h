#pragma once
#include "Camera.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"

// Megakernel based Pathtracer
struct MegaKernel {
	Camera camera;
	float frames_since_camera_moved = 0.0f;

	CUDAModule::Global global_camera_position;
	CUDAModule::Global global_camera_top_left_corner;
	CUDAModule::Global global_camera_x_axis;
	CUDAModule::Global global_camera_y_axis;

	CUDAKernel kernel;

	void init(const char * scene_name, unsigned frame_buffer_handle);

	void update(float delta, const unsigned char * keys);
	void render();
};
