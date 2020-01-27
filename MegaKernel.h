#pragma once
#include "Pathtracer.h"

// Megakernel based Pathtracer
struct MegaKernel : Pathtracer {
	CUDAModule::Global global_camera_position;
	CUDAModule::Global global_camera_top_left_corner;
	CUDAModule::Global global_camera_x_axis;
	CUDAModule::Global global_camera_y_axis;

	CUDAKernel kernel;

	void init(const char * scene_name, unsigned frame_buffer_handle);

	void update(float delta, const unsigned char * keys);
	void render();
};
