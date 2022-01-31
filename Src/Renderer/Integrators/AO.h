#pragma once

#include "Renderer/Integrators/Integrator.h"

struct TraceBufferAO {
	CUDAVector3_SoA origin;
	CUDAVector3_SoA direction;

	CUDAMemory::Ptr<float4> hits;

	CUDAMemory::Ptr<int> pixel_index;

	inline void init(int buffer_size) {
		origin   .init(buffer_size);
		direction.init(buffer_size);
		hits        = CUDAMemory::malloc<float4>(buffer_size);
		pixel_index = CUDAMemory::malloc<int>   (buffer_size);
	}

	inline void free() {
		origin.free();
		direction.free();
		CUDAMemory::free(hits);
		CUDAMemory::free(pixel_index);
	}
};

struct ShadowRayBufferAO {
	CUDAVector3_SoA ray_origin;
	CUDAVector3_SoA ray_direction;

	CUDAMemory::Ptr<float> max_distance;
	CUDAMemory::Ptr<int>   pixel_index;

	inline void init(int buffer_size) {
		ray_origin   .init(buffer_size);
		ray_direction.init(buffer_size);

		max_distance = CUDAMemory::malloc<float>(buffer_size);
		pixel_index  = CUDAMemory::malloc<int>  (buffer_size);
	}

	inline void free() {
		ray_origin.free();
		ray_direction.free();

		CUDAMemory::free(max_distance);
		CUDAMemory::free(pixel_index);
	}
};

struct BufferSizesAO {
	int trace;
	int shadow;

	int rays_retired;
	int rays_retired_shadow;

	void reset(int batch_size) {
		trace               = batch_size;
		shadow              = 0;
		rays_retired        = 0;
		rays_retired_shadow = 0;
	}
};

struct AO final : Integrator {
	CUDAKernel kernel_generate;
	CUDAKernel kernel_trace_bvh2;
	CUDAKernel kernel_trace_bvh4;
	CUDAKernel kernel_trace_bvh8;
	CUDAKernel kernel_ambient_occlusion;
	CUDAKernel kernel_trace_shadow_bvh2;
	CUDAKernel kernel_trace_shadow_bvh4;
	CUDAKernel kernel_trace_shadow_bvh8;

	CUDAKernel * kernel_trace        = nullptr;
	CUDAKernel * kernel_trace_shadow = nullptr;

	CUDAKernel kernel_accumulate;

	TraceBufferAO     ray_buffer_trace;
	ShadowRayBufferAO ray_buffer_shadow;

	BufferSizesAO * pinned_buffer_sizes = nullptr;

	CUDAEvent::Desc event_desc_primary;
	CUDAEvent::Desc event_desc_trace;
	CUDAEvent::Desc event_desc_ambient_occlusion;
	CUDAEvent::Desc event_desc_shadow_trace;
	CUDAEvent::Desc event_desc_accumulate;
	CUDAEvent::Desc event_desc_end;

	float ao_radius = 1.0f;

	AO(unsigned frame_buffer_handle, int width, int height, Scene & scene) : Integrator(scene) {
		cuda_init(frame_buffer_handle, width, height);
	}

	void cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) override;
	void cuda_free() override;

	void init_module();
	void init_events();

	void resize_init(unsigned frame_buffer_handle, int width, int height) override; // Part of resize that initializes new size
	void resize_free()                                                    override; // Part of resize that cleans up old size

	void update(float delta) override;
	void render()            override;

	void render_gui() override;
};
