#pragma once
#include <vector>

#include "CUDA/CUDAModule.h"
#include "CUDA/CUDAKernel.h"
#include "CUDA/CUDAMemory.h"
#include "CUDA/CUDAEvent.h"

#include "Assets/Material.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/SBVHBuilder.h"
#include "BVH/Builders/QBVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

#include "Scene.h"

// Mirror CUDA vector types
struct alignas(8)  float2 { float x, y; };
struct             float3 { float x, y, z; };
struct alignas(16) float4 { float x, y, z, w; };

struct CUDAVector3_SoA {
	CUDAMemory::Ptr<float> x;
	CUDAMemory::Ptr<float> y;
	CUDAMemory::Ptr<float> z;

	inline void init(int buffer_size) {
		x = CUDAMemory::malloc<float>(buffer_size);
		y = CUDAMemory::malloc<float>(buffer_size);
		z = CUDAMemory::malloc<float>(buffer_size);
	}

	inline void free() {
		CUDAMemory::free(x);
		CUDAMemory::free(y);
		CUDAMemory::free(z);
	}
};

struct TraceBuffer {
	CUDAVector3_SoA origin;
	CUDAVector3_SoA direction;

#if ENABLE_MIPMAPPING
	CUDAMemory::Ptr<float2> cone;
#endif
	CUDAMemory::Ptr<float4> hits;

	CUDAMemory::Ptr<int> pixel_index_and_last_material;
	CUDAVector3_SoA      throughput;

	CUDAMemory::Ptr<float> last_pdf;

	inline void init(int buffer_size) {
		origin   .init(buffer_size);
		direction.init(buffer_size);

#if ENABLE_MIPMAPPING
		cone = CUDAMemory::malloc<float2>(buffer_size);
#endif
		hits = CUDAMemory::malloc<float4>(buffer_size);

		pixel_index_and_last_material = CUDAMemory::malloc<int>(buffer_size);
		throughput.init(buffer_size);

		last_pdf = CUDAMemory::malloc<float>(buffer_size);
	}

	inline void free() {
		origin.free();
		direction.free();

#if ENABLE_MIPMAPPING
		CUDAMemory::free(cone);
#endif
		CUDAMemory::free(hits);

		CUDAMemory::free(pixel_index_and_last_material);
		throughput.free();

		CUDAMemory::free(last_pdf);
	}
};

struct MaterialBuffer {
	CUDAVector3_SoA direction;
	
#if ENABLE_MIPMAPPING
	CUDAMemory::Ptr<float2> cone;
#endif
	CUDAMemory::Ptr<float4> hits;

	CUDAMemory::Ptr<int> pixel_index;
	CUDAVector3_SoA      throughput;

	inline void init(int buffer_size) {
		direction.init(buffer_size);
		
#if ENABLE_MIPMAPPING
		cone = CUDAMemory::malloc<float2>(buffer_size);
#endif
		hits = CUDAMemory::malloc<float4>(buffer_size);

		pixel_index = CUDAMemory::malloc<int>(buffer_size);
		throughput.init(buffer_size);
	}

	inline void free() {
		direction.free();

#if ENABLE_MIPMAPPING
		CUDAMemory::free(cone);
#endif
		CUDAMemory::free(hits);

		CUDAMemory::free(pixel_index);
		throughput.free();
	}
};

struct ShadowRayBuffer {
	CUDAVector3_SoA ray_origin;
	CUDAVector3_SoA ray_direction;

	CUDAMemory::Ptr<float> max_distance;

	CUDAMemory::Ptr<int> pixel_index;
	CUDAVector3_SoA      illumination;

	inline void init(int buffer_size) {
		ray_origin   .init(buffer_size);
		ray_direction.init(buffer_size);

		max_distance = CUDAMemory::malloc<float>(buffer_size);

		pixel_index = CUDAMemory::malloc<int>(buffer_size);
		illumination.init(buffer_size);
	}

	inline void free() {
		ray_origin.free();
		ray_direction.free();

		CUDAMemory::free(max_distance);

		CUDAMemory::free(pixel_index);
		illumination.free();
	}
};

struct BufferSizes {
	int trace     [MAX_BOUNCES];
	int diffuse   [MAX_BOUNCES];
	int dielectric[MAX_BOUNCES];
	int glossy    [MAX_BOUNCES];
	int shadow    [MAX_BOUNCES];

	int rays_retired       [MAX_BOUNCES];
	int rays_retired_shadow[MAX_BOUNCES];

	void reset(int batch_size) {
		memset(this, 0, sizeof(BufferSizes));
		trace[0] = batch_size;
	}
};

struct Pathtracer {
	Scene scene;

	bool scene_invalidated     = true;
	bool materials_invalidated = true;
	bool camera_invalidated    = true;
	
	enum struct PixelQueryStatus {
		INACTIVE,
		PENDING,
		OUTPUT_READY
	} pixel_query_status = PixelQueryStatus::INACTIVE;

	int frames_accumulated = -1;

	Settings settings;
	bool     settings_changed = true;

	PixelQuery pixel_query = { INVALID, INVALID, INVALID, INVALID };

	int * reverse_indices;

	int * mesh_data_bvh_offsets;
	int * mesh_data_triangle_offsets;

	void init(int mesh_count, char const ** mesh_names, char const * sky_name, unsigned frame_buffer_handle);

	void cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height);
	void cuda_free();

	void resize_init(unsigned frame_buffer_handle, int width, int height); // Part of resize that initializes new size
	void resize_free();                                                    // Part of resize that cleans up old size

	void update(float delta);
	void render();

	void set_pixel_query(int x, int y);

private:
	int screen_width;
	int screen_height;
	int screen_pitch;

	int pixel_count;
	int batch_size;
	
	BVH        tlas_raw;
	BVHBuilder tlas_bvh_builder;
	BVHType    tlas;

#if BVH_TYPE == BVH_QBVH
	QBVHBuilder tlas_converter;
#elif BVH_TYPE == BVH_CWBVH
	CWBVHBuilder tlas_converter;
#endif
	
	CUDAModule cuda_module;

	CUstream stream_memset;

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
	CUsurfObject       surf_accumulator;
	
	TraceBuffer     ray_buffer_trace;
	MaterialBuffer  ray_buffer_shade_diffuse;
	MaterialBuffer  ray_buffer_shade_dielectric_and_glossy;
	ShadowRayBuffer ray_buffer_shadow;

	CUDAModule::Global global_ray_buffer_shade_diffuse;
	CUDAModule::Global global_ray_buffer_shade_dielectric_and_glossy;
	CUDAModule::Global global_ray_buffer_shadow; 

	BufferSizes * pinned_buffer_sizes;

	CUDAModule::Global global_camera;
	CUDAModule::Global global_buffer_sizes;
	CUDAModule::Global global_settings;
	CUDAModule::Global global_svgf_data;

	CUDAModule::Global global_pixel_query;
	
	CUarray array_gbuffer_normal_and_depth;
	CUarray array_gbuffer_mesh_id_and_triangle_id;
	CUarray array_gbuffer_screen_position_prev;

	CUsurfObject surf_gbuffer_normal_and_depth;
	CUsurfObject surf_gbuffer_mesh_id_and_triangle_id;
	CUsurfObject surf_gbuffer_screen_position_prev;

	CUDAMemory::Ptr<float4> ptr_frame_buffer_albedo;
	CUDAMemory::Ptr<float4> ptr_frame_buffer_moment;

	CUDAMemory::Ptr<float4> ptr_direct;
	CUDAMemory::Ptr<float4> ptr_indirect;
	CUDAMemory::Ptr<float4> ptr_direct_alt;
	CUDAMemory::Ptr<float4> ptr_indirect_alt;

	CUDAMemory::Ptr<int>    ptr_history_length;
	CUDAMemory::Ptr<float4> ptr_history_direct;
	CUDAMemory::Ptr<float4> ptr_history_indirect; 
	CUDAMemory::Ptr<float4> ptr_history_moment;
	CUDAMemory::Ptr<float4> ptr_history_normal_and_depth;

	CUDAMemory::Ptr<float4> ptr_taa_frame_prev;
	CUDAMemory::Ptr<float4> ptr_taa_frame_curr;
	
	struct Matrix3x4 {
		float cells[12];
	};

	int       * pinned_mesh_bvh_root_indices;
	Matrix3x4 * pinned_mesh_transforms;
	Matrix3x4 * pinned_mesh_transforms_inv;
	Matrix3x4 * pinned_mesh_transforms_prev;
	int       * pinned_light_mesh_transform_indices;
	float     * pinned_light_mesh_area_scaled;

	struct CUDATexture {
		CUtexObject texture;
		float2 size;
	};

	CUDATexture      * textures;
	CUmipmappedArray * texture_arrays;

	CUDAMemory::Ptr<Material>    ptr_materials;
	CUDAMemory::Ptr<CUDATexture> ptr_textures;
	
	struct CUDATriangle {
		Vector3 position_0;
		Vector3 position_edge_1;
		Vector3 position_edge_2;

		Vector3 normal_0;
		Vector3 normal_edge_1;
		Vector3 normal_edge_2;

		Vector2 tex_coord_0;
		Vector2 tex_coord_edge_1;
		Vector2 tex_coord_edge_2;
	};
	
	CUDAMemory::Ptr<CUDATriangle> ptr_triangles;
	CUDAMemory::Ptr<int>          ptr_triangle_material_ids;
	
	CUDAMemory::Ptr<BVHNodeType> ptr_bvh_nodes;
	CUDAMemory::Ptr<int>         ptr_mesh_bvh_root_indices;
	CUDAMemory::Ptr<Matrix3x4>   ptr_mesh_transforms;
	CUDAMemory::Ptr<Matrix3x4>   ptr_mesh_transforms_inv;
	CUDAMemory::Ptr<Matrix3x4>   ptr_mesh_transforms_prev;

	CUDAMemory::Ptr<int>   ptr_light_indices;
	CUDAMemory::Ptr<float> ptr_light_areas_cumulative;

	CUDAMemory::Ptr<float> ptr_light_mesh_power_unscaled;
	CUDAMemory::Ptr<int>   ptr_light_mesh_triangle_count;
	CUDAMemory::Ptr<int>   ptr_light_mesh_triangle_first_index;

	CUDAMemory::Ptr<float> ptr_light_mesh_power_scaled;
	CUDAMemory::Ptr<int>   ptr_light_mesh_transform_indices;
	
	CUDAModule::Global global_lights_total_power;

	CUDAMemory::Ptr<Vector3> ptr_sky_data;

	CUDAMemory::Ptr<unsigned char> ptr_sobol_256spp_256d;
	CUDAMemory::Ptr<unsigned char> ptr_scrambling_tile;
	CUDAMemory::Ptr<unsigned char> ptr_ranking_tile;
	
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

	void calc_light_power();

	void build_tlas();
};
