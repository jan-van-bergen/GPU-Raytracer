#pragma once
#include <vector>

#include "Camera.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"
#include "CUDAMemory.h"
#include "CUDAEvent.h"

#include "GBuffer.h"
#include "Shader.h"

// Mirror CUDA vector types
struct alignas(8) float2 { float x, y;       };
struct            float3 { float x, y, z;    };
struct            float4 { float x, y, z, w; };

struct Pathtracer {
	Camera camera;
	int frames_since_camera_moved = -1;
	
	// Settings
	bool settings_changed = true;

	bool enable_rasterization    = true;
	bool enable_svgf             = false;
	bool enable_spatial_variance = true;
	bool enable_taa              = true;
	bool enable_albedo           = true;

	struct SVGFSettings {
		float alpha_colour = 0.2f;
		float alpha_moment = 0.2f;

		int atrous_iterations = 6;

		float sigma_z     = 1.0f;
		float sigma_l_inv = 1.0f / 4.0f;
	} svgf_settings;
	
	std::vector<const CUDAEvent *> events;

private:
	GBuffer gbuffer;

	CUDAModule module;

	CUDAKernel kernel_primary;

	CUDAKernel kernel_generate;
	CUDAKernel kernel_trace;
	CUDAKernel kernel_sort;
	CUDAKernel kernel_shade_diffuse;
	CUDAKernel kernel_shade_dielectric;
	CUDAKernel kernel_shade_glossy;
	CUDAKernel kernel_shadow_trace;
	CUDAKernel kernel_shadow_connect;

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

	CUDAMemory::Ptr<float4> ptr_direct;
	CUDAMemory::Ptr<float4> ptr_indirect;
	CUDAMemory::Ptr<float4> ptr_direct_alt;
	CUDAMemory::Ptr<float4> ptr_indirect_alt;

	// Timing Events
	CUDAEvent event_primary;
	CUDAEvent event_trace[NUM_BOUNCES];
	CUDAEvent event_sort [NUM_BOUNCES];
	CUDAEvent event_shade_diffuse   [NUM_BOUNCES];
	CUDAEvent event_shade_dielectric[NUM_BOUNCES];
	CUDAEvent event_shade_glossy    [NUM_BOUNCES];
	CUDAEvent event_shadow_trace  [NUM_BOUNCES];
	CUDAEvent event_shadow_connect[NUM_BOUNCES];
	CUDAEvent event_svgf_temporal;
	CUDAEvent event_svgf_variance;
	CUDAEvent event_svgf_atrous[MAX_ATROUS_ITERATIONS];
	CUDAEvent event_svgf_finalize;
	CUDAEvent event_taa;
	CUDAEvent event_accumulate;
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
