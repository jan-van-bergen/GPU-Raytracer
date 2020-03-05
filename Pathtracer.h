#pragma once
#include "Camera.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"

#include "Shader.h"

struct Pathtracer {
private:
	Camera camera;
	int frames_since_camera_moved = -1;

	CUstream memcpy_stream;

	CUDAModule module;
	CUDAKernel kernel_primary;
	CUDAKernel kernel_generate;
	CUDAKernel kernel_extend;
	CUDAKernel kernel_shade_diffuse;
	CUDAKernel kernel_shade_dielectric;
	CUDAKernel kernel_shade_glossy;
	CUDAKernel kernel_connect;
	CUDAKernel kernel_accumulate;

	CUDAModule::Global global_buffer_sizes;

	int vertex_count;
	Shader shader;

	GLuint uniform_view_projection;

public:
	void init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle);

	void update(float delta, const unsigned char * keys);
	void render();
};
