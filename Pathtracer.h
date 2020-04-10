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
	bool enable_taa           = true;
	bool enable_albedo        = true;

	struct SVGFSettings {
		float alpha_colour = 0.2f;
		float alpha_moment = 0.2f;

		float sigma_z     =   1.0f;
		float sigma_n     = 128.0f;
		float sigma_l_inv = 1.0f / 40.0f;
	} svgf_settings;

	// Course profile timings
	float time_primary;
	float time_extend[NUM_BOUNCES];
	float time_shade_diffuse   [NUM_BOUNCES];
	float time_shade_dielectric[NUM_BOUNCES];
	float time_shade_glossy    [NUM_BOUNCES];
	float time_connect[NUM_BOUNCES];
	float time_svgf_temporal;
	float time_svgf_atrous[ATROUS_ITERATIONS];
	float time_svgf_finalize;
	float time_taa;

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

	CUDAKernel kernel_taa;
	CUDAKernel kernel_taa_finalize;

	CUDAKernel kernel_accumulate;

	CUDAModule::Global global_buffer_sizes;

	CUDAModule::Global global_svgf_settings;

	int vertex_count;
	Shader shader;

	GLuint uniform_view_projection;
	GLuint uniform_view_projection_prev;

	GLuint uniform_jitter;

	CUDAMemory::Ptr<float> ptr_direct;
	CUDAMemory::Ptr<float> ptr_indirect;
	CUDAMemory::Ptr<float> ptr_direct_alt;
	CUDAMemory::Ptr<float> ptr_indirect_alt;

	// Timing Events
	CUDAEvent event_primary;
	CUDAEvent event_extend[NUM_BOUNCES];
	CUDAEvent event_shade_diffuse   [NUM_BOUNCES];
	CUDAEvent event_shade_dielectric[NUM_BOUNCES];
	CUDAEvent event_shade_glossy    [NUM_BOUNCES];
	CUDAEvent event_connect[NUM_BOUNCES];
	CUDAEvent event_svgf_temporal;
	CUDAEvent event_svgf_atrous[ATROUS_ITERATIONS];
	CUDAEvent event_svgf_finalize;
	CUDAEvent event_taa;
	CUDAEvent event_end;

	bool scene_has_diffuse    = false;
	bool scene_has_dielectric = false;
	bool scene_has_glossy     = false;
	bool scene_has_lights     = false;

public:
	void init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle);

	void update(float delta, const unsigned char * keys);
	void render();
};
