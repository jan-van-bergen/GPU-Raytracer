#include "AO.h"

#include <Imgui/imgui.h>

void AO::cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) {
	init_module();
	init_globals();

	pinned_buffer_sizes = CUDAMemory::malloc_pinned<BufferSizesAO>();
	pinned_buffer_sizes->reset(BATCH_SIZE);

	global_buffer_sizes = cuda_module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	resize_init(frame_buffer_handle, screen_width, screen_height);

	init_geometry();
	init_rng();
	init_events();

	ray_buffer_trace .init(BATCH_SIZE);
	ray_buffer_shadow.init(BATCH_SIZE);
	cuda_module.get_global("ray_buffer_trace") .set_value(ray_buffer_trace);
	cuda_module.get_global("ray_buffer_shadow").set_value(ray_buffer_shadow);

	scene.camera.update(0.0f);
	scene.update(0.0f);

	scene.has_diffuse    = false;
	scene.has_plastic    = false;
	scene.has_dielectric = false;
	scene.has_conductor  = false;
	scene.has_lights     = false;

	invalidated_scene      = true;
	invalidated_materials  = true;
	invalidated_mediums    = true;
	invalidated_gpu_config = true;
	invalidated_aovs       = true;

	size_t bytes_available = CUDAContext::get_available_memory();
	size_t bytes_allocated = CUDAContext::total_memory - bytes_available;
	IO::print("CUDA Memory allocated: {} KB ({} MB)\n"_sv,   bytes_allocated >> 10, bytes_allocated >> 20);
	IO::print("CUDA Memory free:      {} KB ({} MB)\n\n"_sv, bytes_available >> 10, bytes_available >> 20);
}

void AO::cuda_free() {
	Integrator::cuda_free();

	free_geometry();
	free_rng();

	CUDAMemory::free_pinned(pinned_buffer_sizes);

	ray_buffer_trace .free();
	ray_buffer_shadow.free();
}

void AO::init_module() {
	cuda_module.init("AO"_sv, "Src/CUDA/AO.cu"_sv, CUDAContext::compute_capability, MAX_REGISTERS);

	kernel_generate         .init(&cuda_module, "kernel_generate");
	kernel_trace_bvh2       .init(&cuda_module, "kernel_trace_bvh2");
	kernel_trace_bvh4       .init(&cuda_module, "kernel_trace_bvh4");
	kernel_trace_bvh8       .init(&cuda_module, "kernel_trace_bvh8");
	kernel_trace_shadow_bvh2.init(&cuda_module, "kernel_trace_shadow_bvh2");
	kernel_trace_shadow_bvh4.init(&cuda_module, "kernel_trace_shadow_bvh4");
	kernel_trace_shadow_bvh8.init(&cuda_module, "kernel_trace_shadow_bvh8");
	kernel_ambient_occlusion.init(&cuda_module, "kernel_ambient_occlusion");
	kernel_accumulate       .init(&cuda_module, "kernel_accumulate");

	switch (cpu_config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: kernel_trace = &kernel_trace_bvh2; kernel_trace_shadow = &kernel_trace_shadow_bvh2; break;
		case BVHType::BVH4: kernel_trace = &kernel_trace_bvh4; kernel_trace_shadow = &kernel_trace_shadow_bvh4; break;
		case BVHType::BVH8: kernel_trace = &kernel_trace_bvh8; kernel_trace_shadow = &kernel_trace_shadow_bvh8; break;
		default: ASSERT_UNREACHABLE();
	}

	kernel_generate         .set_block_dim(256, 1, 1);
	kernel_ambient_occlusion.set_block_dim(256, 1, 1);
	kernel_accumulate.occupancy_max_block_size_2d();

	// BVH8 uses a stack of int2's (8 bytes)
	// Other BVH's use a stack of ints (4 bytes)
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_bvh2);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_bvh4);
	kernel_trace_calc_grid_and_block_size<8>(kernel_trace_bvh8);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_shadow_bvh2);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_shadow_bvh4);
	kernel_trace_calc_grid_and_block_size<8>(kernel_trace_shadow_bvh8);
}

void AO::init_events() {
	int display_order = 0;
	event_desc_primary = { display_order++, "Primary"_sv, "Primary"_sv };

	event_desc_trace             = CUDAEvent::Desc { display_order, "Bounce 0"_sv, "Trace"_sv };
	event_desc_ambient_occlusion = CUDAEvent::Desc { display_order, "Bounce 0"_sv, "AO"_sv };
	event_desc_shadow_trace      = CUDAEvent::Desc { display_order, "Bounce 0"_sv, "Shadow"_sv };
	display_order++;

	event_desc_accumulate  = CUDAEvent::Desc { display_order, "Post"_sv, "Accumulate"_sv };
	display_order++;

	event_desc_end = CUDAEvent::Desc { display_order, "END"_sv, "END"_sv };
}

void AO::resize_init(unsigned frame_buffer_handle, int width, int height) {
	screen_width  = width;
	screen_height = height;
	screen_pitch  = Math::round_up(width, WARP_SIZE);

	pixel_count = width * height;

	cuda_module.get_global("screen_width") .set_value(screen_width);
	cuda_module.get_global("screen_pitch") .set_value(screen_pitch);
	cuda_module.get_global("screen_height").set_value(screen_height);

	init_aovs();
	aov_enable(AOVType::RADIANCE);
	cuda_module.get_global("aovs").set_value(aovs);

	// Set Accumulator to a CUDA resource mapping of the GL frame buffer texture
	resource_accumulator = CUDAMemory::resource_register(frame_buffer_handle, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
	surf_accumulator     = CUDAMemory::create_surface(CUDAMemory::resource_get_array(resource_accumulator));
	cuda_module.get_global("accumulator").set_value(surf_accumulator);

	kernel_generate         .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_generate         .block_dim_x), 1, 1);
	kernel_ambient_occlusion.set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_ambient_occlusion.block_dim_x), 1, 1);
	kernel_accumulate       .set_grid_dim(screen_pitch / kernel_accumulate.block_dim_x, Math::divide_round_up(height, kernel_accumulate.block_dim_y), 1);

	scene.camera.resize(width, height);
	invalidated_camera = true;

	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(Math::min(BATCH_SIZE, pixel_count));
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	sample_index = 0;
}

void AO::resize_free() {
	CUDACALL(cuStreamSynchronize(memory_stream));

	free_aovs();

	CUDAMemory::resource_unregister(resource_accumulator);
	CUDAMemory::free_surface(surf_accumulator);
}

void AO::update(float delta, Allocator * frame_allocator) {
	if (invalidated_scene) {
		sample_index = 0;
	}

	Integrator::update(delta, frame_allocator);
}

void AO::render() {
	event_pool.reset();

	CUDACALL(cuStreamSynchronize(memory_stream));

	int pixels_left = pixel_count;
	int batch_size  = Math::min(BATCH_SIZE, pixel_count);

	// Render in batches of BATCH_SIZE pixels at a time
	while (pixels_left > 0) {
		int pixel_offset = pixel_count - pixels_left;
		int pixel_count  = Math::min(batch_size, pixels_left);

		// Generate primary Rays from the current Camera orientation
		event_pool.record(event_desc_primary);
		kernel_generate.execute(sample_index, pixel_offset, pixel_count);

		event_pool.record(event_desc_trace);
		kernel_trace->execute();

		event_pool.record(event_desc_ambient_occlusion);
		kernel_ambient_occlusion.execute(sample_index, ao_radius);

		event_pool.record(event_desc_shadow_trace);
		kernel_trace_shadow->execute();

		pixels_left -= batch_size;

		if (pixels_left > 0) {
			// Set buffer sizes to appropriate pixel count for next Batch
			pinned_buffer_sizes->reset(Math::min(batch_size, pixels_left));
			global_buffer_sizes.set_value(*pinned_buffer_sizes);
		}
	}

	event_pool.record(event_desc_accumulate);
	kernel_accumulate.execute(float(sample_index));

	event_pool.record(event_desc_end);

	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(batch_size);
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	aovs_clear_to_zero();

	// If a pixel query was previously pending, it has just been resolved in the current frame
	if (pixel_query_status == PixelQueryStatus::PENDING) {
		pixel_query_status =  PixelQueryStatus::OUTPUT_READY;
	}
}

void AO::render_gui() {
	if (ImGui::CollapsingHeader("Integrator", ImGuiTreeNodeFlags_DefaultOpen)) {
		invalidated_gpu_config |= ImGui::SliderFloat("AO Radius", &ao_radius, 0.0001f, 2.0f);
	}

	if (ImGui::CollapsingHeader("Auxilary AOVs", ImGuiTreeNodeFlags_DefaultOpen)) {
		invalidated_aovs |= aov_render_gui_checkbox(AOVType::NORMAL,   "Normal");
		invalidated_aovs |= aov_render_gui_checkbox(AOVType::POSITION, "Position");
	}
}
