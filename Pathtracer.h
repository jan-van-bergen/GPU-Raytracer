#pragma once
#include "Camera.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"

struct Pathtracer {
private:
	Camera camera;
	int frames_since_camera_moved = -1;

	CUstream stream;

	CUDAModule module;
	CUDAKernel kernel_generate;
	CUDAKernel kernel_extend;
	CUDAKernel kernel_shade_diffuse;
	CUDAKernel kernel_shade_dielectric;
	CUDAKernel kernel_shade_glossy;
	CUDAKernel kernel_connect;
	CUDAKernel kernel_accumulate;

	CUDAModule::Global global_buffer_sizes;

public:
	void init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle);

	void update(float delta, const unsigned char * keys);
	void render();
};
