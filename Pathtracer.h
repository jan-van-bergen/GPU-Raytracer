#pragma once
#include "Camera.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"

struct Pathtracer {
	Camera camera;
	float frames_since_camera_moved = 0.0f;

	CUDAKernel kernel_generate;
	CUDAKernel kernel_extend;
	CUDAKernel kernel_shade;
	CUDAKernel kernel_connect;
	CUDAKernel kernel_accumulate;

	CUDAModule::Global global_ray_buffer_0;
	CUDAModule::Global global_ray_buffer_1;

	CUDAModule::Global global_N_ext;
	CUDAModule::Global global_N_shadow;

	void init(const char * scene_name, unsigned frame_buffer_handle);

	void update(float delta, const unsigned char * keys);
	void render();
};
