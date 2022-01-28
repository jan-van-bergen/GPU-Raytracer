#pragma once
#include "Renderer/Integrators/Integrator.h"
#include "Renderer/Material.h"

struct TraceBuffer {
	CUDAVector3_SoA ray_origin;
	CUDAVector3_SoA ray_direction;

	CUDAMemory::Ptr<float4> hits;

	CUDAMemory::Ptr<float2> cone;

	CUDAMemory::Ptr<int> medium;

	CUDAMemory::Ptr<int> pixel_index_and_flags;
	CUDAVector3_SoA      throughput;

	CUDAMemory::Ptr<float> last_pdf;

	void init(int buffer_size) {
		ray_origin   .init(buffer_size);
		ray_direction.init(buffer_size);

		medium = CUDAMemory::malloc<int>(buffer_size);

		cone = CUDAMemory::malloc<float2>(buffer_size);
		hits = CUDAMemory::malloc<float4>(buffer_size);

		pixel_index_and_flags = CUDAMemory::malloc<int>(buffer_size);
		throughput.init(buffer_size);

		last_pdf = CUDAMemory::malloc<float>(buffer_size);
	}

	void free() {
		ray_origin.free();
		ray_direction.free();

		CUDAMemory::free(medium);

		CUDAMemory::free(cone);
		CUDAMemory::free(hits);

		CUDAMemory::free(pixel_index_and_flags);
		throughput.free();

		CUDAMemory::free(last_pdf);
	}
};

struct MaterialBuffer {
	CUDAVector3_SoA direction;

	CUDAMemory::Ptr<int> medium;

	CUDAMemory::Ptr<float2> cone;
	CUDAMemory::Ptr<float4> hits;

	CUDAMemory::Ptr<int> pixel_index;
	CUDAVector3_SoA      throughput;

	void init(int buffer_size) {
		direction.init(buffer_size);

		medium = CUDAMemory::malloc<int>(buffer_size);

		cone = CUDAMemory::malloc<float2>(buffer_size);
		hits = CUDAMemory::malloc<float4>(buffer_size);

		pixel_index = CUDAMemory::malloc<int>(buffer_size);
		throughput.init(buffer_size);
	}

	void free() {
		direction.free();

		CUDAMemory::free(medium);

		CUDAMemory::free(cone);
		CUDAMemory::free(hits);

		CUDAMemory::free(pixel_index);
		throughput.free();
	}
};

struct ShadowRayBuffer {
	CUDAVector3_SoA ray_origin;
	CUDAVector3_SoA ray_direction;

	CUDAMemory::Ptr<float>  max_distance;
	CUDAMemory::Ptr<float4> illumination_and_pixel_index;

	void init(int buffer_size) {
		ray_origin   .init(buffer_size);
		ray_direction.init(buffer_size);

		max_distance                 = CUDAMemory::malloc<float> (buffer_size);
		illumination_and_pixel_index = CUDAMemory::malloc<float4>(buffer_size);
	}

	void free() {
		ray_origin.free();
		ray_direction.free();

		CUDAMemory::free(max_distance);
		CUDAMemory::free(illumination_and_pixel_index);
	}
};
struct BufferSizes {
	int trace     [MAX_BOUNCES];
	int diffuse   [MAX_BOUNCES];
	int plastic   [MAX_BOUNCES];
	int dielectric[MAX_BOUNCES];
	int conductor [MAX_BOUNCES];
	int shadow    [MAX_BOUNCES];

	int rays_retired       [MAX_BOUNCES];
	int rays_retired_shadow[MAX_BOUNCES];

	void reset(int batch_size) {
		memset(trace,               0, sizeof(trace));
		memset(diffuse,             0, sizeof(diffuse));
		memset(plastic,             0, sizeof(plastic));
		memset(dielectric,          0, sizeof(dielectric));
		memset(conductor,           0, sizeof(conductor));
		memset(shadow,              0, sizeof(shadow));
		memset(rays_retired,        0, sizeof(rays_retired));
		memset(rays_retired_shadow, 0, sizeof(rays_retired_shadow));

		trace[0] = batch_size;
	}
};

struct Pathtracer final : Integrator {
	CUDAKernel kernel_generate;
	CUDAKernel kernel_trace_bvh2;
	CUDAKernel kernel_trace_bvh4;
	CUDAKernel kernel_trace_bvh8;
	CUDAKernel kernel_sort;
	CUDAKernel kernel_material_diffuse;
	CUDAKernel kernel_material_plastic;
	CUDAKernel kernel_material_dielectric;
	CUDAKernel kernel_material_conductor;
	CUDAKernel kernel_trace_shadow_bvh2;
	CUDAKernel kernel_trace_shadow_bvh4;
	CUDAKernel kernel_trace_shadow_bvh8;

	CUDAKernel * kernel_trace        = nullptr;
	CUDAKernel * kernel_trace_shadow = nullptr;

	CUDAKernel kernel_svgf_reproject;
	CUDAKernel kernel_svgf_variance;
	CUDAKernel kernel_svgf_atrous;
	CUDAKernel kernel_svgf_finalize;

	CUDAKernel kernel_taa;
	CUDAKernel kernel_taa_finalize;

	CUDAKernel kernel_accumulate;

	TraceBuffer     ray_buffer_trace_0;
	TraceBuffer     ray_buffer_trace_1;
	ShadowRayBuffer ray_buffer_shadow;

	Array<MaterialBuffer>               material_ray_buffers;
	CUDAMemory::Ptr<MaterialBuffer> ptr_material_ray_buffers;

	CUDAModule::Global global_ray_buffer_shadow;

	BufferSizes * pinned_buffer_sizes = nullptr;

	CUDAModule::Global global_svgf_data;

	CUarray array_gbuffer_normal_and_depth;
	CUarray array_gbuffer_mesh_id_and_triangle_id;
	CUarray array_gbuffer_screen_position_prev;

	CUsurfObject surf_gbuffer_normal_and_depth;
	CUsurfObject surf_gbuffer_mesh_id_and_triangle_id;
	CUsurfObject surf_gbuffer_screen_position_prev;

	CUDAMemory::Ptr<float4> ptr_frame_buffer_albedo;
	CUDAMemory::Ptr<float4> ptr_frame_buffer_moment;

	CUDAMemory::Ptr<float4> ptr_frame_buffer_direct;
	CUDAMemory::Ptr<float4> ptr_frame_buffer_indirect;
	CUDAMemory::Ptr<float4> ptr_frame_buffer_direct_alt;
	CUDAMemory::Ptr<float4> ptr_frame_buffer_indirect_alt;

	CUDAMemory::Ptr<int>    ptr_history_length;
	CUDAMemory::Ptr<float4> ptr_history_direct;
	CUDAMemory::Ptr<float4> ptr_history_indirect;
	CUDAMemory::Ptr<float4> ptr_history_moment;
	CUDAMemory::Ptr<float4> ptr_history_normal_and_depth;

	CUDAMemory::Ptr<float4> ptr_taa_frame_prev;
	CUDAMemory::Ptr<float4> ptr_taa_frame_curr;

	Array<double> light_mesh_probabilites; // Scratch memory used to compute pinned_light_mesh_prob_alias

	CUDAModule::Global global_lights_total_weight;

	CUDAMemory::Ptr<int>       ptr_light_indices;
	CUDAMemory::Ptr<ProbAlias> ptr_light_prob_alias;

	CUDAMemory::Ptr<ProbAlias> ptr_light_mesh_prob_alias;
	CUDAMemory::Ptr<int2>      ptr_light_mesh_first_index_and_triangle_count;
	CUDAMemory::Ptr<int>       ptr_light_mesh_transform_index;

	// Timing Events
	CUDAEvent::Desc event_desc_primary;
	CUDAEvent::Desc event_desc_trace[MAX_BOUNCES];
	CUDAEvent::Desc event_desc_sort [MAX_BOUNCES];
	CUDAEvent::Desc event_desc_material_diffuse   [MAX_BOUNCES];
	CUDAEvent::Desc event_desc_material_plastic   [MAX_BOUNCES];
	CUDAEvent::Desc event_desc_material_dielectric[MAX_BOUNCES];
	CUDAEvent::Desc event_desc_material_conductor [MAX_BOUNCES];
	CUDAEvent::Desc event_desc_shadow_trace[MAX_BOUNCES];
	CUDAEvent::Desc event_desc_svgf_reproject;
	CUDAEvent::Desc event_desc_svgf_variance;
	CUDAEvent::Desc event_desc_svgf_atrous[MAX_ATROUS_ITERATIONS];
	CUDAEvent::Desc event_desc_svgf_finalize;
	CUDAEvent::Desc event_desc_taa;
	CUDAEvent::Desc event_desc_reconstruct;
	CUDAEvent::Desc event_desc_accumulate;
	CUDAEvent::Desc event_desc_end;

	Pathtracer(unsigned frame_buffer_handle, int width, int height, Scene & scene) : Integrator(scene) {
		cuda_init(frame_buffer_handle, width, height);
	}

	void cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) override;
	void cuda_free() override;

	void init_module();
	void init_events();

	void resize_init(unsigned frame_buffer_handle, int width, int height) override; // Part of resize that initializes new size
	void resize_free()                                                    override; // Part of resize that cleans up old size

	void svgf_init();
	void svgf_free();

	void update(float delta) override;
	void render()            override;

	void render_gui() override;

	void calc_light_power();
	void calc_light_mesh_weights();
};
