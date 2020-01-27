#pragma once
#include "Pathtracer.h"

// Wavefront based Pathtracer
struct Wavefront : Pathtracer {
	CUDAKernel kernel_generate;
	CUDAKernel kernel_extend;
	CUDAKernel kernel_shade_diffuse;
	CUDAKernel kernel_shade_dielectric;
	CUDAKernel kernel_shade_glossy;
	CUDAKernel kernel_connect;
	CUDAKernel kernel_accumulate;

	CUDAModule::Global global_N_extend;
	CUDAModule::Global global_N_diffuse;
	CUDAModule::Global global_N_dielectric;
	CUDAModule::Global global_N_glossy;
	CUDAModule::Global global_N_shadow;

	void init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle);

	void render();
};
