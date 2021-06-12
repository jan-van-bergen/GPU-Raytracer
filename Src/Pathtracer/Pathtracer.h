#pragma once
#include <vector>

#include "CUDA/CUDAModule.h"
#include "CUDA/CUDAKernel.h"
#include "CUDA/CUDAMemory.h"
#include "CUDA/CUDAEvent.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/SBVHBuilder.h"
#include "BVH/Builders/QBVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

#include "Scene.h"

// Mirror CUDA vector types
struct alignas(8)  float2 { float x, y; };
struct             float3 { float x, y, z; };
struct alignas(16) float4 { float x, y, z, w; };

struct Pathtracer {
	Scene scene;

	bool camera_invalidated = true;
	bool first_frame_after_stopped_updating = true;

	bool read_pixel_query = false;

	int frames_accumulated = -1;

	Settings settings;
	bool     settings_changed = true;

	PixelQuery       pixel_query        = { -1, -1 };
	PixelQueryAnswer pixel_query_answer = { -1, -1 };

	void init(int mesh_count, char const ** mesh_names, char const * sky_name, unsigned frame_buffer_handle);

	void resize_init(unsigned frame_buffer_handle, int width, int height); // Part of resize that initializes new size
	void resize_free();                                                    // Part of resize that cleans up old size

	void update(float delta);
	void render();

private:
	int screen_width;
	int screen_height;
	int screen_pitch;

	int pixel_count;
	int batch_size;
	
	CUDAModule module;

	CUDAKernel kernel_generate;
	CUDAKernel kernel_trace;
	CUDAKernel kernel_sort;
	CUDAKernel kernel_shade_diffuse;
	CUDAKernel kernel_shade_dielectric;
	CUDAKernel kernel_shade_glossy;
	CUDAKernel kernel_trace_shadow;

	CUDAKernel kernel_svgf_reproject;
	CUDAKernel kernel_svgf_variance;
	CUDAKernel kernel_svgf_atrous;
	CUDAKernel kernel_svgf_finalize;

	CUDAKernel kernel_taa;
	CUDAKernel kernel_taa_finalize;

	CUDAKernel kernel_accumulate;

	CUgraphicsResource resource_accumulator;

	CUDAModule::Global global_camera;
	CUDAModule::Global global_buffer_sizes;
	CUDAModule::Global global_settings;
	CUDAModule::Global global_svgf_data;

	CUDAModule::Global global_pixel_query;
	CUDAModule::Global global_pixel_query_answer;
	
	CUDAMemory::Ptr<float4> ptr_direct;
	CUDAMemory::Ptr<float4> ptr_indirect;
	CUDAMemory::Ptr<float4> ptr_direct_alt;
	CUDAMemory::Ptr<float4> ptr_indirect_alt;

	// Timing Events
	CUDAEvent::Info event_info_primary;
	CUDAEvent::Info event_info_trace[MAX_BOUNCES];
	CUDAEvent::Info event_info_sort [MAX_BOUNCES];
	CUDAEvent::Info event_info_shade_diffuse   [MAX_BOUNCES];
	CUDAEvent::Info event_info_shade_dielectric[MAX_BOUNCES];
	CUDAEvent::Info event_info_shade_glossy    [MAX_BOUNCES];
	CUDAEvent::Info event_info_shadow_trace[MAX_BOUNCES];
	CUDAEvent::Info event_info_svgf_reproject;
	CUDAEvent::Info event_info_svgf_variance;
	CUDAEvent::Info event_info_svgf_atrous[MAX_ATROUS_ITERATIONS];
	CUDAEvent::Info event_info_svgf_finalize;
	CUDAEvent::Info event_info_taa;
	CUDAEvent::Info event_info_reconstruct;
	CUDAEvent::Info event_info_accumulate;
	CUDAEvent::Info event_info_end;

	BVH        tlas_raw;
	BVHBuilder tlas_bvh_builder;
	BVHType    tlas;

#if BVH_TYPE == BVH_QBVH
	QBVHBuilder tlas_converter;
#elif BVH_TYPE == BVH_CWBVH
	CWBVHBuilder tlas_converter;
#endif
	
	int * mesh_data_bvh_offsets;

	struct Matrix3x4 {
		float cells[12];
	};

	int       * pinned_mesh_bvh_root_indices;
	Matrix3x4 * pinned_mesh_transforms;
	Matrix3x4 * pinned_mesh_transforms_inv;
	Matrix3x4 * pinned_mesh_transforms_prev;
	int       * pinned_light_mesh_transform_indices;
	float     * pinned_light_mesh_area_scaled;

	CUDAMemory::Ptr<BVHNodeType> ptr_bvh_nodes;
	CUDAMemory::Ptr<int>         ptr_mesh_bvh_root_indices;
	CUDAMemory::Ptr<Matrix3x4>   ptr_mesh_transforms;
	CUDAMemory::Ptr<Matrix3x4>   ptr_mesh_transforms_inv;
	CUDAMemory::Ptr<Matrix3x4>   ptr_mesh_transforms_prev;

	CUDAMemory::Ptr<float> ptr_light_total_area;
	CUDAMemory::Ptr<float> ptr_light_mesh_area_scaled;
	CUDAMemory::Ptr<int>   ptr_light_mesh_transform_indices;

	void build_tlas();
};
