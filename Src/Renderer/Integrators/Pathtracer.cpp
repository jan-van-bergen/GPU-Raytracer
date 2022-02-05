#include "Pathtracer.h"

#include <Imgui/imgui.h>

void Pathtracer::cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) {
	init_module();
	init_globals();

	pinned_buffer_sizes = CUDAMemory::malloc_pinned<BufferSizes>();
	pinned_buffer_sizes->reset(BATCH_SIZE);

	global_buffer_sizes = cuda_module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	resize_init(frame_buffer_handle, screen_width, screen_height);

	init_materials();
	init_geometry();
	init_sky();
	init_rng();
	init_events();

	ray_buffer_trace_0.init(BATCH_SIZE);
	ray_buffer_trace_1.init(BATCH_SIZE);
	cuda_module.get_global("ray_buffer_trace_0").set_value(ray_buffer_trace_0);
	cuda_module.get_global("ray_buffer_trace_1").set_value(ray_buffer_trace_1);

	global_ray_buffer_shadow = cuda_module.get_global("ray_buffer_shadow");

	global_svgf_data = cuda_module.get_global("svgf_data");

	global_lights_total_weight = cuda_module.get_global("lights_total_weight");
	global_lights_total_weight.set_value(0.0f);

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

void Pathtracer::cuda_free() {
	Integrator::cuda_free();

	free_materials();
	free_geometry();
	free_sky();
	free_rng();

	CUDAMemory::free_pinned(pinned_buffer_sizes);

	if (scene.has_lights) {
		CUDAMemory::free(ptr_light_indices);
		CUDAMemory::free(ptr_light_prob_alias);

		CUDAMemory::free(ptr_light_mesh_prob_alias);
		CUDAMemory::free(ptr_light_mesh_first_index_and_triangle_count);
		CUDAMemory::free(ptr_light_mesh_transform_index);

		ray_buffer_shadow.free();
	}

	ray_buffer_trace_0.free();
	ray_buffer_trace_1.free();

	for (size_t i = 0; i < material_ray_buffers.size(); i++) {
		material_ray_buffers[i].free();
	}
	material_ray_buffers.clear();

	CUDAMemory::free(ptr_material_ray_buffers);
}

void Pathtracer::init_module() {
	cuda_module.init("Pathtracer"_sv, "Src/CUDA/Pathtracer.cu"_sv, CUDAContext::compute_capability, MAX_REGISTERS);

	kernel_generate           .init(&cuda_module, "kernel_generate");
	kernel_trace_bvh2         .init(&cuda_module, "kernel_trace_bvh2");
	kernel_trace_bvh4         .init(&cuda_module, "kernel_trace_bvh4");
	kernel_trace_bvh8         .init(&cuda_module, "kernel_trace_bvh8");
	kernel_sort               .init(&cuda_module, "kernel_sort");
	kernel_material_diffuse   .init(&cuda_module, "kernel_material_diffuse");
	kernel_material_plastic   .init(&cuda_module, "kernel_material_plastic");
	kernel_material_dielectric.init(&cuda_module, "kernel_material_dielectric");
	kernel_material_conductor .init(&cuda_module, "kernel_material_conductor");
	kernel_trace_shadow_bvh2  .init(&cuda_module, "kernel_trace_shadow_bvh2");
	kernel_trace_shadow_bvh4  .init(&cuda_module, "kernel_trace_shadow_bvh4");
	kernel_trace_shadow_bvh8  .init(&cuda_module, "kernel_trace_shadow_bvh8");
	kernel_svgf_reproject     .init(&cuda_module, "kernel_svgf_reproject");
	kernel_svgf_variance      .init(&cuda_module, "kernel_svgf_variance");
	kernel_svgf_atrous        .init(&cuda_module, "kernel_svgf_atrous");
	kernel_svgf_finalize      .init(&cuda_module, "kernel_svgf_finalize");
	kernel_taa                .init(&cuda_module, "kernel_taa");
	kernel_taa_finalize       .init(&cuda_module, "kernel_taa_finalize");
	kernel_accumulate         .init(&cuda_module, "kernel_accumulate");

	switch (cpu_config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: kernel_trace = &kernel_trace_bvh2; kernel_trace_shadow = &kernel_trace_shadow_bvh2; break;
		case BVHType::BVH4: kernel_trace = &kernel_trace_bvh4; kernel_trace_shadow = &kernel_trace_shadow_bvh4; break;
		case BVHType::BVH8: kernel_trace = &kernel_trace_bvh8; kernel_trace_shadow = &kernel_trace_shadow_bvh8; break;
		default: ASSERT(false);
	}

	// Set Block dimensions for all Kernels
	kernel_generate           .set_block_dim(256, 1, 1);
	kernel_sort               .set_block_dim(256, 1, 1);
	kernel_material_diffuse   .set_block_dim(256, 1, 1);
	kernel_material_plastic   .set_block_dim(256, 1, 1);
	kernel_material_dielectric.set_block_dim(256, 1, 1);
	kernel_material_conductor .set_block_dim(256, 1, 1);

	kernel_svgf_reproject.occupancy_max_block_size_2d();
	kernel_svgf_variance .occupancy_max_block_size_2d();
	kernel_svgf_atrous   .occupancy_max_block_size_2d();
	kernel_svgf_finalize .occupancy_max_block_size_2d();
	kernel_taa           .occupancy_max_block_size_2d();
	kernel_taa_finalize  .occupancy_max_block_size_2d();
	kernel_accumulate    .occupancy_max_block_size_2d();

	// BVH8 uses a stack of int2's (8 bytes)
	// Other BVH's use a stack of ints (4 bytes)
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_bvh2);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_bvh4);
	kernel_trace_calc_grid_and_block_size<8>(kernel_trace_bvh8);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_shadow_bvh2);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_shadow_bvh4);
	kernel_trace_calc_grid_and_block_size<8>(kernel_trace_shadow_bvh8);
}

void Pathtracer::init_events() {
	int display_order = 0;
	event_desc_primary = { display_order++, "Primary"_sv, "Primary"_sv };

	for (int i = 0; i < MAX_BOUNCES; i++) {
		String category = Format().format("Bounce {}"_sv, i);

		event_desc_trace              [i] = CUDAEvent::Desc { display_order, category, "Trace"_sv };
		event_desc_sort               [i] = CUDAEvent::Desc { display_order, category, "Sort"_sv };
		event_desc_material_diffuse   [i] = CUDAEvent::Desc { display_order, category, "Diffuse"_sv };
		event_desc_material_plastic   [i] = CUDAEvent::Desc { display_order, category, "Plastic"_sv };
		event_desc_material_dielectric[i] = CUDAEvent::Desc { display_order, category, "Dielectric"_sv };
		event_desc_material_conductor [i] = CUDAEvent::Desc { display_order, category, "Conductor"_sv };
		event_desc_shadow_trace       [i] = CUDAEvent::Desc { display_order, category, "Shadow"_sv };

		display_order++;
	}

	event_desc_svgf_reproject = CUDAEvent::Desc { display_order, "SVGF"_sv, "Reproject"_sv };
	event_desc_svgf_variance  = CUDAEvent::Desc { display_order, "SVGF"_sv, "Variance"_sv };

	for (int i = 0; i < MAX_ATROUS_ITERATIONS; i++) {
		event_desc_svgf_atrous[i] = CUDAEvent::Desc { display_order, "SVGF"_sv, Format().format("A Trous {}"_sv, i) };
	}
	event_desc_svgf_finalize = CUDAEvent::Desc { display_order++, "SVGF"_sv, "Finalize"_sv };

	event_desc_taa         = CUDAEvent::Desc { display_order, "Post"_sv, "TAA"_sv };
	event_desc_reconstruct = CUDAEvent::Desc { display_order, "Post"_sv, "Reconstruct"_sv };
	event_desc_accumulate  = CUDAEvent::Desc { display_order, "Post"_sv, "Accumulate"_sv };

	event_desc_end = CUDAEvent::Desc { ++display_order, "END"_sv, "END"_sv };
}

void Pathtracer::resize_init(unsigned frame_buffer_handle, int width, int height) {
	screen_width  = width;
	screen_height = height;
	screen_pitch  = Math::round_up(width, WARP_SIZE);

	pixel_count = width * height;

	cuda_module.get_global("screen_width") .set_value(screen_width);
	cuda_module.get_global("screen_pitch") .set_value(screen_pitch);
	cuda_module.get_global("screen_height").set_value(screen_height);

	// Create Frame Buffers
	init_aovs();
	aov_enable(AOVType::RADIANCE);

	// Set Accumulator to a CUDA resource mapping of the GL frame buffer texture
	resource_accumulator = CUDAMemory::resource_register(frame_buffer_handle, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
	surf_accumulator     = CUDAMemory::create_surface(CUDAMemory::resource_get_array(resource_accumulator));
	cuda_module.get_global("accumulator").set_value(surf_accumulator);

	// Set Grid dimensions for screen size dependent Kernels
	kernel_svgf_reproject.set_grid_dim(screen_pitch / kernel_svgf_reproject.block_dim_x, Math::divide_round_up(height, kernel_svgf_reproject.block_dim_y), 1);
	kernel_svgf_variance .set_grid_dim(screen_pitch / kernel_svgf_variance .block_dim_x, Math::divide_round_up(height, kernel_svgf_variance .block_dim_y), 1);
	kernel_svgf_atrous   .set_grid_dim(screen_pitch / kernel_svgf_atrous   .block_dim_x, Math::divide_round_up(height, kernel_svgf_atrous   .block_dim_y), 1);
	kernel_svgf_finalize .set_grid_dim(screen_pitch / kernel_svgf_finalize .block_dim_x, Math::divide_round_up(height, kernel_svgf_finalize .block_dim_y), 1);
	kernel_taa           .set_grid_dim(screen_pitch / kernel_taa           .block_dim_x, Math::divide_round_up(height, kernel_taa           .block_dim_y), 1);
	kernel_taa_finalize  .set_grid_dim(screen_pitch / kernel_taa_finalize  .block_dim_x, Math::divide_round_up(height, kernel_taa_finalize  .block_dim_y), 1);
	kernel_accumulate    .set_grid_dim(screen_pitch / kernel_accumulate    .block_dim_x, Math::divide_round_up(height, kernel_accumulate    .block_dim_y), 1);

	kernel_generate           .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_generate           .block_dim_x), 1, 1);
	kernel_sort               .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_sort               .block_dim_x), 1, 1);
	kernel_material_diffuse   .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_material_diffuse   .block_dim_x), 1, 1);
	kernel_material_plastic   .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_material_plastic   .block_dim_x), 1, 1);
	kernel_material_dielectric.set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_material_dielectric.block_dim_x), 1, 1);
	kernel_material_conductor .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_material_conductor .block_dim_x), 1, 1);

	scene.camera.resize(width, height);
	invalidated_camera = true;

	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(Math::min(BATCH_SIZE, pixel_count));
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	sample_index = 0;

	if (gpu_config.enable_svgf) svgf_init();
}

void Pathtracer::resize_free() {
	CUDACALL(cuStreamSynchronize(memory_stream));

	free_aovs();

	CUDAMemory::resource_unregister(resource_accumulator);
	CUDAMemory::free_surface(surf_accumulator);

	if (gpu_config.enable_svgf) {
		svgf_free();
	}
}

void Pathtracer::svgf_init() {
	// GBuffers
	array_gbuffer_normal_and_depth        = CUDAMemory::create_array(screen_pitch, screen_height, 4, CU_AD_FORMAT_FLOAT);
	array_gbuffer_mesh_id_and_triangle_id = CUDAMemory::create_array(screen_pitch, screen_height, 2, CU_AD_FORMAT_SIGNED_INT32);
	array_gbuffer_screen_position_prev    = CUDAMemory::create_array(screen_pitch, screen_height, 2, CU_AD_FORMAT_FLOAT);

	surf_gbuffer_normal_and_depth        = CUDAMemory::create_surface(array_gbuffer_normal_and_depth);
	surf_gbuffer_mesh_id_and_triangle_id = CUDAMemory::create_surface(array_gbuffer_mesh_id_and_triangle_id);
	surf_gbuffer_screen_position_prev    = CUDAMemory::create_surface(array_gbuffer_screen_position_prev);

	cuda_module.get_global("gbuffer_normal_and_depth")       .set_value(surf_gbuffer_normal_and_depth);
	cuda_module.get_global("gbuffer_mesh_id_and_triangle_id").set_value(surf_gbuffer_mesh_id_and_triangle_id);
	cuda_module.get_global("gbuffer_screen_position_prev")   .set_value(surf_gbuffer_screen_position_prev);

	// Frame Buffers
	aov_enable(AOVType::RADIANCE_DIRECT);
	aov_enable(AOVType::RADIANCE_INDIRECT);
	aov_enable(AOVType::ALBEDO);

	ptr_frame_buffer_moment = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	cuda_module.get_global("frame_buffer_moment").set_value(ptr_frame_buffer_moment);

	// History Buffers
	ptr_history_length           = CUDAMemory::malloc<int>   (screen_pitch * screen_height);
	ptr_history_direct           = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_history_indirect         = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_history_moment           = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_history_normal_and_depth = CUDAMemory::malloc<float4>(screen_pitch * screen_height);

	cuda_module.get_global("history_length")          .set_value(ptr_history_length);
	cuda_module.get_global("history_direct")          .set_value(ptr_history_direct);
	cuda_module.get_global("history_indirect")        .set_value(ptr_history_indirect);
	cuda_module.get_global("history_moment")          .set_value(ptr_history_moment);
	cuda_module.get_global("history_normal_and_depth").set_value(ptr_history_normal_and_depth);

	// Frame Buffers for Temporal Anti-Aliasing
	ptr_taa_frame_prev = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_taa_frame_curr = CUDAMemory::malloc<float4>(screen_pitch * screen_height);

	cuda_module.get_global("taa_frame_prev").set_value(ptr_taa_frame_prev);
	cuda_module.get_global("taa_frame_curr").set_value(ptr_taa_frame_curr);
}

void Pathtracer::svgf_free() {
	CUDAMemory::free_array(array_gbuffer_normal_and_depth);
	CUDAMemory::free_array(array_gbuffer_mesh_id_and_triangle_id);
	CUDAMemory::free_array(array_gbuffer_screen_position_prev);

	CUDAMemory::free_surface(surf_gbuffer_normal_and_depth);
	CUDAMemory::free_surface(surf_gbuffer_mesh_id_and_triangle_id);
	CUDAMemory::free_surface(surf_gbuffer_screen_position_prev);

	aov_disable(AOVType::RADIANCE_DIRECT);
	aov_disable(AOVType::RADIANCE_INDIRECT);
	aov_disable(AOVType::ALBEDO);

	CUDAMemory::free(ptr_frame_buffer_moment);

	CUDAMemory::free(ptr_history_length);
	CUDAMemory::free(ptr_history_direct);
	CUDAMemory::free(ptr_history_indirect);
	CUDAMemory::free(ptr_history_moment);
	CUDAMemory::free(ptr_history_normal_and_depth);

	CUDAMemory::free(ptr_taa_frame_prev);
	CUDAMemory::free(ptr_taa_frame_curr);
}

void Pathtracer::calc_light_power() {
	HashMap<int, Array<Mesh *>> mesh_data_used_as_lights;

	int light_mesh_count = 0;

	// For every Mesh, check whether it is a Light based on its Material
	// If so, mark the MeshData it is using as being a Light
	for (int m = 0; m < scene.meshes.size(); m++) {
		Mesh & mesh = scene.meshes[m];
		const Material & material = scene.asset_manager.get_material(mesh.material_handle);

		if (material.type == Material::Type::LIGHT) {
			mesh_data_used_as_lights[mesh.mesh_data_handle.handle].push_back(&mesh);
			light_mesh_count++;
		} else {
			mesh.light.weight = 0.0f;
		}
	}

	struct LightTriangle {
		int    index;
		double area;
	};
	Array<LightTriangle> light_triangles;

	struct LightMeshData {
		size_t first_triangle_index;
		size_t triangle_count;

		double total_area;
	};
	Array<LightMeshData> light_mesh_datas;

	using It = decltype(mesh_data_used_as_lights)::Iterator;

	for (It it = mesh_data_used_as_lights.begin(); it != mesh_data_used_as_lights.end(); ++it) {
		MeshDataHandle  mesh_data_handle = MeshDataHandle { it.get_key() };
		Array<Mesh *> & meshes           = it.get_value();

		const MeshData & mesh_data = scene.asset_manager.get_mesh_data(mesh_data_handle);

		LightMeshData & light_mesh_data = light_mesh_datas.emplace_back();
		light_mesh_data.first_triangle_index = light_triangles.size();
		light_mesh_data.triangle_count = mesh_data.triangles.size();
		light_mesh_data.total_area = 0.0f;

		for (int t = 0; t < mesh_data.triangles.size(); t++) {
			const Triangle & triangle = mesh_data.triangles[t];

			float area = 0.5f * Vector3::length(Vector3::cross(
				triangle.position_1 - triangle.position_0,
				triangle.position_2 - triangle.position_0
			));
			light_triangles.emplace_back(reverse_indices[mesh_data_triangle_offsets[mesh_data_handle.handle] + t], area);
			light_mesh_data.total_area += area;
		}

		for (int m = 0; m < meshes.size(); m++) {
			Mesh * mesh = meshes[m];

			const Material & material = scene.asset_manager.get_material(mesh->material_handle);
			float power = Math::luminance(material.emission.x, material.emission.y, material.emission.z);

			mesh->light.weight               = power * float(light_mesh_data.total_area);
			mesh->light.first_triangle_index = light_mesh_data.first_triangle_index;
			mesh->light.triangle_count       = light_mesh_data.triangle_count;
		}
	}

	if (light_triangles.size() > 0) {
		Array<int>       light_indices      (light_triangles.size());
		Array<double>    light_probabilities(light_triangles.size());
		Array<ProbAlias> light_prob_alias   (light_triangles.size());

		for (int m = 0; m < light_mesh_datas.size(); m++) {
			const LightMeshData & light_mesh_data = light_mesh_datas[m];

			for (int i = light_mesh_data.first_triangle_index; i < light_mesh_data.first_triangle_index + light_mesh_data.triangle_count; i++) {
				light_indices      [i] = light_triangles[i].index;
				light_probabilities[i] = light_triangles[i].area / light_mesh_data.total_area;
			}

			Random::alias_method(
				light_mesh_data.triangle_count,
				light_probabilities.data() + light_mesh_data.first_triangle_index,
				light_prob_alias   .data() + light_mesh_data.first_triangle_index
			);
		}

		ptr_light_indices    = CUDAMemory::malloc(light_indices);
		ptr_light_prob_alias = CUDAMemory::malloc(light_prob_alias);

		cuda_module.get_global("light_indices")   .set_value_async(ptr_light_indices,    memory_stream);
		cuda_module.get_global("light_prob_alias").set_value_async(ptr_light_prob_alias, memory_stream);

		cuda_module.get_global("light_mesh_count").set_value_async(light_mesh_count, memory_stream);

		if (ptr_light_mesh_prob_alias                    .ptr != NULL) CUDAMemory::free(ptr_light_mesh_prob_alias);
		if (ptr_light_mesh_first_index_and_triangle_count.ptr != NULL) CUDAMemory::free(ptr_light_mesh_first_index_and_triangle_count);
		if (ptr_light_mesh_transform_index               .ptr != NULL) CUDAMemory::free(ptr_light_mesh_transform_index);

		// The Device pointers below are only filled in and copied to the GPU once the TLAS is constructed,
		// therefore the scene_invalidated flag is required to be set.
		invalidated_scene = true;

		ptr_light_mesh_prob_alias                     = CUDAMemory::malloc<ProbAlias>(light_mesh_count);
		ptr_light_mesh_first_index_and_triangle_count = CUDAMemory::malloc<int2>     (light_mesh_count);
		ptr_light_mesh_transform_index                = CUDAMemory::malloc<int>      (light_mesh_count);

		cuda_module.get_global("light_mesh_prob_alias")                    .set_value_async(ptr_light_mesh_prob_alias,                     memory_stream);
		cuda_module.get_global("light_mesh_first_index_and_triangle_count").set_value_async(ptr_light_mesh_first_index_and_triangle_count, memory_stream);
		cuda_module.get_global("light_mesh_transform_index")               .set_value_async(ptr_light_mesh_transform_index,                memory_stream);
	}
}

// Construct Top Level Acceleration Structure (TLAS) over the Meshes in the Scene
void Pathtracer::calc_light_mesh_weights() {
	int    light_mesh_count    = 0;
	double lights_total_weight = 0.0;

	light_mesh_probabilites.resize(scene.meshes.size());

	for (int i = 0; i < scene.meshes.size(); i++) {
		const Mesh & mesh = scene.meshes[tlas->indices[i]];

		bool mesh_is_light = mesh.light.weight > 0.0f;
		if (mesh_is_light) {
			int light_index = light_mesh_count++;

			double light_weight_scaled = mesh.light.weight * mesh.scale * mesh.scale;
			lights_total_weight += light_weight_scaled;

			light_mesh_probabilites                         [light_index]   = light_weight_scaled;
			pinned_light_mesh_first_index_and_triangle_count[light_index].x = mesh.light.first_triangle_index;
			pinned_light_mesh_first_index_and_triangle_count[light_index].y = mesh.light.triangle_count;
			pinned_light_mesh_transform_index               [light_index]   = i;
		}
	}

	if (light_mesh_count > 0) {
		for (int i = 0; i < light_mesh_count; i++) {
			light_mesh_probabilites[i] /= lights_total_weight;
		}
		Random::alias_method(light_mesh_count, light_mesh_probabilites.data(), pinned_light_mesh_prob_alias);

		CUDAMemory::memcpy_async(ptr_light_mesh_prob_alias,                     pinned_light_mesh_prob_alias,                     light_mesh_count, memory_stream);
		CUDAMemory::memcpy_async(ptr_light_mesh_first_index_and_triangle_count, pinned_light_mesh_first_index_and_triangle_count, light_mesh_count, memory_stream);
		CUDAMemory::memcpy_async(ptr_light_mesh_transform_index,                pinned_light_mesh_transform_index,                light_mesh_count, memory_stream);
	}

	global_lights_total_weight.set_value_async(float(lights_total_weight), memory_stream);
}

void Pathtracer::update(float delta) {
	if (invalidated_materials) {
		const Array<Material> & materials = scene.asset_manager.materials;

		Array<Material::Type> cuda_material_types(materials.size());
		Array<CUDAMaterial>   cuda_materials     (materials.size());

		for (int i = 0; i < materials.size(); i++) {
			const Material & material = materials[i];

			cuda_material_types[i] = material.type;

			switch (material.type) {
				case Material::Type::LIGHT: {
					cuda_materials[i].light.emission = material.emission;
					break;
				}
				case Material::Type::DIFFUSE: {
					cuda_materials[i].diffuse.diffuse    = material.diffuse;
					cuda_materials[i].diffuse.texture_id = material.texture_id.handle;
					break;
				}
				case Material::Type::PLASTIC: {
					cuda_materials[i].plastic.diffuse    = material.diffuse;
					cuda_materials[i].plastic.texture_id = material.texture_id.handle;
					cuda_materials[i].plastic.roughness  = Math::max(material.linear_roughness * material.linear_roughness, 1e-6f);
					break;
				}
				case Material::Type::DIELECTRIC: {
					cuda_materials[i].dielectric.medium_id = material.medium_handle.handle;
					cuda_materials[i].dielectric.ior       = Math::max(material.index_of_refraction, 1.0001f);
					cuda_materials[i].dielectric.roughness = Math::max(material.linear_roughness * material.linear_roughness, 1e-6f);
					break;
				}
				case Material::Type::CONDUCTOR: {
					cuda_materials[i].conductor.eta       = material.eta;
					cuda_materials[i].conductor.roughness = Math::max(material.linear_roughness * material.linear_roughness, 1e-6f);
					cuda_materials[i].conductor.k         = material.k;
					break;
				}
				default: ASSERT(false);
			}
		}

		CUDAMemory::memcpy_async(ptr_material_types, cuda_material_types.data(), materials.size(), memory_stream);
		CUDAMemory::memcpy_async(ptr_materials,      cuda_materials     .data(), materials.size(), memory_stream);

		bool had_diffuse    = scene.has_diffuse;
		bool had_plastic    = scene.has_plastic;
		bool had_dielectric = scene.has_dielectric;
		bool had_conductor  = scene.has_conductor;
		bool had_lights     = scene.has_lights;

		scene.calc_properties();

		bool material_types_changed =
			(had_diffuse    ^ scene.has_diffuse) |
			(had_plastic    ^ scene.has_plastic) |
			(had_dielectric ^ scene.has_dielectric) |
			(had_conductor  ^ scene.has_conductor);

		if (material_types_changed) {
			int num_different_materials =
				int(scene.has_diffuse) +
				int(scene.has_plastic) +
				int(scene.has_dielectric) +
				int(scene.has_conductor);
			// 2 different Materials can share the same MaterialBuffer, one growing left to right, one growing right to left
			int num_material_buffers_needed = Math::divide_round_up(num_different_materials, 2);

			if (num_material_buffers_needed < material_ray_buffers.size()) {
				// Free MaterialBuffers that are not needed
				for (int i = num_material_buffers_needed; i < material_ray_buffers.size(); i++) {
					material_ray_buffers[i].free();
				}
			} else if (num_material_buffers_needed > material_ray_buffers.size()) {
				// Allocate new required MaterialBuffers
				for (int i = material_ray_buffers.size(); i < num_material_buffers_needed; i++) {
					material_ray_buffers.emplace_back().init(BATCH_SIZE);
				}
			}

			if (ptr_material_ray_buffers.ptr) {
				CUDAMemory::free(ptr_material_ray_buffers);
			}
			ptr_material_ray_buffers = CUDAMemory::malloc(material_ray_buffers);

			int material_buffer_index = 0;

			auto set_material_buffer = [&](const char * global_name) {
				// Two consecutive buffers can share the same underlying allocation, every odd buffer is reversed
				CUDAMemory::Ptr<MaterialBuffer> ptr_buffer = ptr_material_ray_buffers + material_buffer_index / 2;
				bool reversed = material_buffer_index & 1;

				ASSERT((ptr_buffer.ptr & 1) == 0);
				uintptr_t packed = uintptr_t(ptr_buffer.ptr) | uintptr_t(reversed);
				cuda_module.get_global(global_name).set_value_async(packed, memory_stream);

				material_buffer_index++;
			};
			if (scene.has_diffuse)    set_material_buffer("material_buffer_diffuse");
			if (scene.has_plastic)    set_material_buffer("material_buffer_plastic");
			if (scene.has_dielectric) set_material_buffer("material_buffer_dielectric");
			if (scene.has_conductor)  set_material_buffer("material_buffer_conductor");
		}

		// Handle (dis)appearance of Light materials
		bool lights_changed = had_lights ^ scene.has_lights;
		if (lights_changed) {
			if (scene.has_lights) {
				ray_buffer_shadow.init(BATCH_SIZE);

				invalidated_scene = true;
			} else {
				ray_buffer_shadow.free();

				CUDAMemory::free(ptr_light_mesh_prob_alias);
				CUDAMemory::free(ptr_light_mesh_first_index_and_triangle_count);
				CUDAMemory::free(ptr_light_mesh_transform_index);

				global_lights_total_weight.set_value_async(0.0f, memory_stream);

				for (int i = 0; i < scene.meshes.size(); i++) {
					scene.meshes[i].light.weight = 0.0f;
				}
			}

			global_ray_buffer_shadow.set_value_async(ray_buffer_shadow, memory_stream);
		}

		if (had_lights) {
			CUDAMemory::free(ptr_light_indices);
			CUDAMemory::free(ptr_light_prob_alias);
		}
		if (scene.has_lights) {
			calc_light_power();
		}

		sample_index = 0;
		invalidated_materials = false;
	}

	if (invalidated_mediums) {
		size_t medium_count = scene.asset_manager.media.size();
		if (medium_count > 0) {
			Array<CUDAMedium> cuda_mediums(medium_count);

			for (size_t i = 0; i < medium_count; i++) {
				const Medium & medium = scene.asset_manager.media[i];
				medium.get_sigmas(cuda_mediums[i].sigma_a, cuda_mediums[i].sigma_s);
				cuda_mediums[i].g = medium.g;
			}

			CUDAMemory::memcpy_async(ptr_media, cuda_mediums.data(), medium_count, memory_stream);
		}

		sample_index = 0;
		invalidated_mediums = false;
	}

	// Save this here, TLAS will be updated in Integrator::update if invalidated_scene == true,
	// calc_light_mesh_weights() will need to be called AFTER the TLAS has been updated.
	bool invalidated_light_mesh_weights = invalidated_scene;

	if (gpu_config.enable_svgf) {
		struct SVGFData {
			alignas(16) Matrix4 view_projection;
			alignas(16) Matrix4 view_projection_prev;
		} svgf_data;

		svgf_data.view_projection      = scene.camera.view_projection;
		svgf_data.view_projection_prev = scene.camera.view_projection_prev;

		global_svgf_data.set_value_async(svgf_data, memory_stream);
	}

	if (invalidated_aovs) {
		if (gpu_config.enable_svgf && !aov_is_enabled(AOVType::ALBEDO)) {
			aov_enable(AOVType::ALBEDO); // SVGF cannot function without ALBEDO
		}
	}

	Integrator::update(delta);

	if (invalidated_light_mesh_weights) {
		calc_light_mesh_weights();

		// If SVGF is enabled we can handle Scene updates using reprojection,
		// otherwise 'frames_accumulated' needs to be reset in order to avoid ghosting
		if (!gpu_config.enable_svgf) {
			sample_index = 0;
		}
	}
}

void Pathtracer::render() {
	event_pool.reset();

	CUDACALL(cuStreamSynchronize(memory_stream));

	int pixels_left = pixel_count;
	int batch_size  = Math::min(BATCH_SIZE, pixel_count);

	// Render in batches of BATCH_SIZE pixels at a time
	while (pixels_left > 0) {
		int pixel_offset = pixel_count - pixels_left;
		int pixel_count  = Math::min(batch_size, pixels_left);

		event_pool.record(event_desc_primary);

		// Generate primary Rays from the current Camera orientation
		kernel_generate.execute(sample_index, pixel_offset, pixel_count);

		for (int bounce = 0; bounce < gpu_config.num_bounces; bounce++) {
			// Extend all Rays that are still alive to their next Triangle intersection
			event_pool.record(event_desc_trace[bounce]);
			kernel_trace->execute(bounce);

			event_pool.record(event_desc_sort[bounce]);
			kernel_sort.execute(bounce, sample_index);

			// Process the various Material types in different Kernels
			if (scene.has_diffuse) {
				event_pool.record(event_desc_material_diffuse[bounce]);
				kernel_material_diffuse.execute(bounce, sample_index);
			}
			if (scene.has_plastic) {
				event_pool.record(event_desc_material_plastic[bounce]);
				kernel_material_plastic.execute(bounce, sample_index);
			}
			if (scene.has_dielectric) {
				event_pool.record(event_desc_material_dielectric[bounce]);
				kernel_material_dielectric.execute(bounce, sample_index);
			}
			if (scene.has_conductor) {
				event_pool.record(event_desc_material_conductor[bounce]);
				kernel_material_conductor.execute(bounce, sample_index);
			}

			// Trace shadow Rays
			if (scene.has_lights && gpu_config.enable_next_event_estimation) {
				event_pool.record(event_desc_shadow_trace[bounce]);
				kernel_trace_shadow->execute(bounce);
			}
		}

		pixels_left -= batch_size;

		if (pixels_left > 0) {
			// Set buffer sizes to appropriate pixel count for next Batch
			pinned_buffer_sizes->reset(Math::min(batch_size, pixels_left));
			global_buffer_sizes.set_value(*pinned_buffer_sizes);
		}
	}

	if (gpu_config.enable_svgf) {
		// Temporal reprojection + integration
		event_pool.record(event_desc_svgf_reproject);
		kernel_svgf_reproject.execute(sample_index);

		CUdeviceptr direct_in    = get_aov(AOVType::RADIANCE_DIRECT)  .framebuffer.ptr;
		CUdeviceptr indirect_in  = get_aov(AOVType::RADIANCE_INDIRECT).framebuffer.ptr;
		CUdeviceptr direct_out   = get_aov(AOVType::RADIANCE_DIRECT)  .accumulator.ptr;
		CUdeviceptr indirect_out = get_aov(AOVType::RADIANCE_INDIRECT).accumulator.ptr;

		if (gpu_config.enable_spatial_variance) {
			// Estimate Variance spatially
			event_pool.record(event_desc_svgf_variance);
			kernel_svgf_variance.execute(direct_in, indirect_in, direct_out, indirect_out);

			Util::swap(direct_in,   direct_out);
			Util::swap(indirect_in, indirect_out);
		}

		// À-Trous Filter
		for (int i = 0; i < gpu_config.num_atrous_iterations; i++) {
			int step_size = 1 << i;

			event_pool.record(event_desc_svgf_atrous[i]);
			kernel_svgf_atrous.execute(direct_in, indirect_in, direct_out, indirect_out, step_size);

			// Ping-Pong the Frame Buffers
			Util::swap(direct_in,   direct_out);
			Util::swap(indirect_in, indirect_out);
		}

		event_pool.record(event_desc_svgf_finalize);
		kernel_svgf_finalize.execute(direct_in, indirect_in);

		if (gpu_config.enable_taa) {
			event_pool.record(event_desc_taa);

			kernel_taa         .execute(sample_index);
			kernel_taa_finalize.execute();
		}
	} else {
		event_pool.record(event_desc_accumulate);
		kernel_accumulate.execute(float(sample_index));
	}

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

void Pathtracer::render_gui() {
	if (ImGui::CollapsingHeader("Integrator", ImGuiTreeNodeFlags_DefaultOpen)) {
		invalidated_gpu_config |= ImGui::SliderInt("Num Bounces", &gpu_config.num_bounces, 0, MAX_BOUNCES);

		invalidated_gpu_config |= ImGui::Checkbox("NEE", &gpu_config.enable_next_event_estimation);
		invalidated_gpu_config |= ImGui::Checkbox("MIS", &gpu_config.enable_multiple_importance_sampling);

		invalidated_gpu_config |= ImGui::Checkbox("Russian Roulete", &gpu_config.enable_russian_roulette);
	}

	if (ImGui::CollapsingHeader("Auxilary AOVs", ImGuiTreeNodeFlags_DefaultOpen)) {
		invalidated_aovs |= aov_render_gui_checkbox(AOVType::ALBEDO,   "Albedo");
		invalidated_aovs |= aov_render_gui_checkbox(AOVType::NORMAL,   "Normal");
		invalidated_aovs |= aov_render_gui_checkbox(AOVType::POSITION, "Position");
	}

	if (ImGui::CollapsingHeader("SVGF")) {
		if (ImGui::Checkbox("Enable", &gpu_config.enable_svgf)) {
			if (gpu_config.enable_svgf) {
				svgf_init();
			} else {
				svgf_free();
			}
			invalidated_gpu_config = true;
		}

		invalidated_gpu_config |= ImGui::Checkbox("Spatial Variance", &gpu_config.enable_spatial_variance);
		invalidated_gpu_config |= ImGui::Checkbox("TAA",              &gpu_config.enable_taa);

		invalidated_gpu_config |= ImGui::SliderInt("A Trous iterations", &gpu_config.num_atrous_iterations, 0, MAX_ATROUS_ITERATIONS);

		invalidated_gpu_config |= ImGui::SliderFloat("Alpha colour", &gpu_config.alpha_colour, 0.0f, 1.0f);
		invalidated_gpu_config |= ImGui::SliderFloat("Alpha moment", &gpu_config.alpha_moment, 0.0f, 1.0f);
	}
}
