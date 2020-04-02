#pragma once
#include "Camera.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"
#include "CUDAMemory.h"
#include "CUDAEvent.h"

#include "GBuffer.h"
#include "Shader.h"

struct Pathtracer {
	Camera camera;
	int frames_since_camera_moved = -1;
	
	// Settings
	bool enable_rasterization = true;
	bool enable_svgf          = true;
	bool enable_albedo        = true;

	// Course profile timings
	float time_primary;
	float time_extend;
	float time_svgf;

private:
	GBuffer gbuffer;

	CUDAModule module;

	CUDAKernel kernel_primary;

	CUDAKernel kernel_generate;
	CUDAKernel kernel_extend;
	CUDAKernel kernel_shade_diffuse;
	CUDAKernel kernel_shade_dielectric;
	CUDAKernel kernel_shade_glossy;
	CUDAKernel kernel_connect;

	CUDAKernel kernel_svgf_temporal;
	CUDAKernel kernel_svgf_variance;
	CUDAKernel kernel_svgf_atrous;
	CUDAKernel kernel_svgf_finalize;

	CUDAKernel kernel_accumulate;

	CUDAModule::Global global_buffer_sizes;

	CUDAModule::Global global_camera_depth_range;

	int vertex_count;
	Shader shader;

	GLuint uniform_view_projection;
	GLuint uniform_view_projection_prev;

	CUDAMemory::Ptr<float> ptr_direct;
	CUDAMemory::Ptr<float> ptr_indirect;
	CUDAMemory::Ptr<float> ptr_direct_alt;
	CUDAMemory::Ptr<float> ptr_indirect_alt;

	// Timing Events
	CUDAEvent event_primary;
	CUDAEvent event_extend;
	CUDAEvent event_svgf;
	CUDAEvent event_end;

public:
	void init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle);

	void update(float delta, const unsigned char * keys);
	void render();
};
