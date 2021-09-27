#include "Pathtracer.h"

#include <GL/glew.h>

#include "Config.h"

#include "CUDA/CUDAContext.h"

#include "Assets/MeshData.h"
#include "Assets/Material.h"
#include "Assets/MitsubaLoader.h"

#include "Math/Vector4.h"

#include "Util/BlueNoise.h"

#include "Util/Util.h"
#include "Util/HashMap.h"
#include "Util/ScopeTimer.h"

void Pathtracer::init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle, int width, int height) {
	scene.init(scene_name, sky_name);

	cuda_init(frame_buffer_handle, width, height);

	CUDACALL(cuStreamCreate(&stream_memset, CU_STREAM_NON_BLOCKING));
}

template<size_t BVH_STACK_ELEMENT_SIZE>
void kernel_trace_calc_grid_and_block_size(CUDAKernel & kernel) {
	CUoccupancyB2DSize block_size_to_shared_memory = [](int block_size) {
		return size_t(block_size) * SHARED_STACK_SIZE * BVH_STACK_ELEMENT_SIZE;
	};

	int grid, block;
	CUDACALL(cuOccupancyMaxPotentialBlockSize(&grid, &block, kernel.kernel, block_size_to_shared_memory, 0, 0));

	int block_x = WARP_SIZE;
	int block_y = block / WARP_SIZE;

	kernel.set_block_dim(block_x, block_y, 1);
	kernel.set_grid_dim(1, grid, 1);
	kernel.set_shared_memory(block_size_to_shared_memory(block));
};

void Pathtracer::cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) {
	// Init CUDA Module and its Kernels
	cuda_module.init("CUDA_Source/Pathtracer.cu", CUDAContext::compute_capability, MAX_REGISTERS);

	kernel_generate          .init(&cuda_module, "kernel_generate");
	kernel_trace_bvh         .init(&cuda_module, "kernel_trace_bvh");
	kernel_trace_qbvh        .init(&cuda_module, "kernel_trace_qbvh");
	kernel_trace_cwbvh       .init(&cuda_module, "kernel_trace_cwbvh");
	kernel_sort              .init(&cuda_module, "kernel_sort");
	kernel_shade_diffuse     .init(&cuda_module, "kernel_shade_diffuse");
	kernel_shade_dielectric  .init(&cuda_module, "kernel_shade_dielectric");
	kernel_shade_glossy      .init(&cuda_module, "kernel_shade_glossy");
	kernel_trace_shadow_bvh  .init(&cuda_module, "kernel_trace_shadow_bvh");
	kernel_trace_shadow_qbvh .init(&cuda_module, "kernel_trace_shadow_qbvh");
	kernel_trace_shadow_cwbvh.init(&cuda_module, "kernel_trace_shadow_cwbvh");
	kernel_svgf_reproject    .init(&cuda_module, "kernel_svgf_reproject");
	kernel_svgf_variance     .init(&cuda_module, "kernel_svgf_variance");
	kernel_svgf_atrous       .init(&cuda_module, "kernel_svgf_atrous");
	kernel_svgf_finalize     .init(&cuda_module, "kernel_svgf_finalize");
	kernel_taa               .init(&cuda_module, "kernel_taa");
	kernel_taa_finalize      .init(&cuda_module, "kernel_taa_finalize");
	kernel_accumulate        .init(&cuda_module, "kernel_accumulate");

	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH:  kernel_trace = &kernel_trace_bvh;   kernel_trace_shadow = &kernel_trace_shadow_bvh;   break;
		case BVHType::QBVH:  kernel_trace = &kernel_trace_qbvh;  kernel_trace_shadow = &kernel_trace_shadow_qbvh;  break;
		case BVHType::CWBVH: kernel_trace = &kernel_trace_cwbvh; kernel_trace_shadow = &kernel_trace_shadow_cwbvh; break;
		default: abort();
	}

	// Set Block dimensions for all Kernels
	kernel_svgf_reproject.occupancy_max_block_size_2d();
	kernel_svgf_variance .occupancy_max_block_size_2d();
	kernel_svgf_atrous   .occupancy_max_block_size_2d();
	kernel_svgf_finalize .occupancy_max_block_size_2d();
	kernel_taa           .occupancy_max_block_size_2d();
	kernel_taa_finalize  .occupancy_max_block_size_2d();
	kernel_accumulate    .occupancy_max_block_size_2d();

	kernel_generate        .set_block_dim(WARP_SIZE * 8, 1, 1);
	kernel_sort            .set_block_dim(WARP_SIZE * 8, 1, 1);
	kernel_shade_diffuse   .set_block_dim(WARP_SIZE * 8, 1, 1);
	kernel_shade_dielectric.set_block_dim(WARP_SIZE * 8, 1, 1);
	kernel_shade_glossy    .set_block_dim(WARP_SIZE * 8, 1, 1);

	// CWBVH uses a stack of int2's (8 bytes)
	// Other BVH's use a stack of ints (4 bytes)
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_bvh);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_qbvh);
	kernel_trace_calc_grid_and_block_size<8>(kernel_trace_cwbvh);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_shadow_bvh);
	kernel_trace_calc_grid_and_block_size<4>(kernel_trace_shadow_qbvh);
	kernel_trace_calc_grid_and_block_size<8>(kernel_trace_shadow_cwbvh);

//	printf("\nConfiguration picked for Tracing kernels:\n    Block Size: %i x %i\n    Grid Size:  %i\n\n", block_x, block_y, grid);

	pinned_buffer_sizes = CUDAMemory::malloc_pinned<BufferSizes>();
	pinned_buffer_sizes->reset(BATCH_SIZE);

	global_buffer_sizes = cuda_module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	resize_init(frame_buffer_handle, screen_width, screen_height);

	// Set global Material table
	ptr_material_types = CUDAMemory::malloc<Material::Type>(scene.asset_manager.materials.size());
	ptr_materials      = CUDAMemory::malloc<CUDAMaterial>  (scene.asset_manager.materials.size());
	cuda_module.get_global("material_types").set_value(ptr_material_types);
	cuda_module.get_global("materials")     .set_value(ptr_materials);

	scene.asset_manager.wait_until_loaded();

	// Set global Texture table
	int texture_count = scene.asset_manager.textures.size();
	if (texture_count > 0) {
		textures       = new CUDATexture     [texture_count];
		texture_arrays = new CUmipmappedArray[texture_count];

		// Get maximum anisotropy from OpenGL
		int max_aniso; glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_aniso);

		for (int i = 0; i < texture_count; i++) {
			const Texture & texture = scene.asset_manager.textures[i];

			// Create mipmapped CUDA array
			texture_arrays[i] = CUDAMemory::create_array_mipmap(
				texture.width,
				texture.height,
				texture.channels,
				texture.get_cuda_array_format(),
				texture.mip_levels
			);

			// Upload each level of the mipmap
			for (int level = 0; level < texture.mip_levels; level++) {
				CUarray level_array;
				CUDACALL(cuMipmappedArrayGetLevel(&level_array, texture_arrays[i], level));

				int level_width_in_bytes = texture.get_width_in_bytes(level);
				int level_height         = Math::max(texture.height >> level, 1);

				CUDAMemory::copy_array(level_array, level_width_in_bytes, level_height, texture.data + texture.mip_offsets[level]);
			}

			// Describe the Array to read from
			CUDA_RESOURCE_DESC res_desc = { };
			res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
			res_desc.res.mipmap.hMipmappedArray = texture_arrays[i];

			// Describe how to sample the Texture
			CUDA_TEXTURE_DESC tex_desc = { };
			tex_desc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.addressMode[2] = CUaddress_mode::CU_TR_ADDRESS_MODE_CLAMP;
			tex_desc.filterMode       = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
			tex_desc.mipmapFilterMode = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
			tex_desc.mipmapLevelBias = 0;
			tex_desc.maxAnisotropy = max_aniso;
			tex_desc.minMipmapLevelClamp = 0;
			tex_desc.maxMipmapLevelClamp = texture.mip_levels - 1;
			tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;

			// Describe the Texture View
			CUDA_RESOURCE_VIEW_DESC view_desc = { };
			view_desc.format = texture.get_cuda_resource_view_format();
			view_desc.width  = texture.get_cuda_resource_view_width();
			view_desc.height = texture.get_cuda_resource_view_height();
			view_desc.firstMipmapLevel = 0;
			view_desc.lastMipmapLevel  = texture.mip_levels - 1;

			CUDACALL(cuTexObjectCreate(&textures[i].texture, &res_desc, &tex_desc, &view_desc));

			textures[i].size.x = texture.width;
			textures[i].size.y = texture.height;
		}

		ptr_textures = CUDAMemory::malloc(textures, texture_count);
		cuda_module.get_global("textures").set_value(ptr_textures);
	}

	int mesh_data_count = scene.asset_manager.mesh_datas.size();

	mesh_data_bvh_offsets      = new int[mesh_data_count];
	mesh_data_triangle_offsets = new int[mesh_data_count];
	int * mesh_data_index_offsets = MALLOCA(int, mesh_data_count);

	int aggregated_bvh_node_count = 2 * scene.meshes.size(); // Reserve 2 times Mesh count for TLAS
	int aggregated_triangle_count = 0;
	int aggregated_index_count    = 0;

	for (int i = 0; i < mesh_data_count; i++) {
		mesh_data_bvh_offsets     [i] = aggregated_bvh_node_count;
		mesh_data_triangle_offsets[i] = aggregated_triangle_count;
		mesh_data_index_offsets   [i] = aggregated_index_count;

		aggregated_bvh_node_count += scene.asset_manager.mesh_datas[i].bvh.node_count;
		aggregated_triangle_count += scene.asset_manager.mesh_datas[i].triangle_count;
		aggregated_index_count    += scene.asset_manager.mesh_datas[i].bvh.index_count;
	}

	char         * aggregated_bvh_nodes = new char[aggregated_bvh_node_count * BVH::node_size()];
	CUDATriangle * aggregated_triangles = new CUDATriangle[aggregated_index_count];

	reverse_indices = new int[aggregated_triangle_count];

	for (int m = 0; m < mesh_data_count; m++) {
		const MeshData & mesh_data = scene.asset_manager.mesh_datas[m];

		mesh_data.bvh.aggregate(aggregated_bvh_nodes, mesh_data_index_offsets[m], mesh_data_bvh_offsets[m]);

		for (int i = 0; i < mesh_data.bvh.index_count; i++) {
			int index = mesh_data.bvh.indices[i];
			const Triangle & triangle = mesh_data.triangles[index];

			aggregated_triangles[mesh_data_triangle_offsets[m] + i].position_0      = triangle.position_0;
			aggregated_triangles[mesh_data_triangle_offsets[m] + i].position_edge_1 = triangle.position_1 - triangle.position_0;
			aggregated_triangles[mesh_data_triangle_offsets[m] + i].position_edge_2 = triangle.position_2 - triangle.position_0;

			aggregated_triangles[mesh_data_triangle_offsets[m] + i].normal_0      = triangle.normal_0;
			aggregated_triangles[mesh_data_triangle_offsets[m] + i].normal_edge_1 = triangle.normal_1 - triangle.normal_0;
			aggregated_triangles[mesh_data_triangle_offsets[m] + i].normal_edge_2 = triangle.normal_2 - triangle.normal_0;

			aggregated_triangles[mesh_data_triangle_offsets[m] + i].tex_coord_0      = triangle.tex_coord_0;
			aggregated_triangles[mesh_data_triangle_offsets[m] + i].tex_coord_edge_1 = triangle.tex_coord_1 - triangle.tex_coord_0;
			aggregated_triangles[mesh_data_triangle_offsets[m] + i].tex_coord_edge_2 = triangle.tex_coord_2 - triangle.tex_coord_0;

			reverse_indices[mesh_data_index_offsets[m] + index] = mesh_data_triangle_offsets[m] + i;
		}
	}

	FREEA(mesh_data_index_offsets);

	pinned_mesh_bvh_root_indices                     = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());
	pinned_mesh_material_ids                         = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());
	pinned_mesh_transforms                           = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_mesh_transforms_inv                       = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_mesh_transforms_prev                      = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_light_mesh_prob_alias                     = CUDAMemory::malloc_pinned<ProbAlias>(scene.meshes.size());
	pinned_light_mesh_first_index_and_triangle_count = CUDAMemory::malloc_pinned<int2>     (scene.meshes.size());
	pinned_light_mesh_transform_index                = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());

	light_mesh_probabilites = new double[scene.meshes.size()];

	ptr_mesh_bvh_root_indices = CUDAMemory::malloc<int>      (scene.meshes.size());
	ptr_mesh_material_ids     = CUDAMemory::malloc<int>      (scene.meshes.size());
	ptr_mesh_transforms       = CUDAMemory::malloc<Matrix3x4>(scene.meshes.size());
	ptr_mesh_transforms_inv   = CUDAMemory::malloc<Matrix3x4>(scene.meshes.size());
	ptr_mesh_transforms_prev  = CUDAMemory::malloc<Matrix3x4>(scene.meshes.size());

	cuda_module.get_global("mesh_bvh_root_indices").set_value(ptr_mesh_bvh_root_indices);
	cuda_module.get_global("mesh_material_ids")    .set_value(ptr_mesh_material_ids);
	cuda_module.get_global("mesh_transforms")      .set_value(ptr_mesh_transforms);
	cuda_module.get_global("mesh_transforms_inv")  .set_value(ptr_mesh_transforms_inv);
	cuda_module.get_global("mesh_transforms_prev") .set_value(ptr_mesh_transforms_prev);

	ptr_bvh_nodes = CUDAMemory::malloc<char>(aggregated_bvh_nodes, aggregated_bvh_node_count * BVH::node_size());
	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH:  cuda_module.get_global("bvh_nodes")  .set_value(ptr_bvh_nodes); break;
		case BVHType::QBVH:  cuda_module.get_global("qbvh_nodes") .set_value(ptr_bvh_nodes); break;
		case BVHType::CWBVH: cuda_module.get_global("cwbvh_nodes").set_value(ptr_bvh_nodes); break;
	}

	tlas_bvh_builder.init(&tlas_raw, scene.meshes.size());

	tlas_raw.index_count = scene.meshes.size();
	tlas_raw.node_count  = scene.meshes.size() * 2;
	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH:  tlas = tlas_raw; break;
		case BVHType::QBVH:  tlas_converter_qbvh .init(&tlas, tlas_raw); break;
		case BVHType::CWBVH: tlas_converter_cwbvh.init(&tlas, tlas_raw); break;
		default: abort();
	}

	ptr_triangles = CUDAMemory::malloc(aggregated_triangles, aggregated_index_count);

	cuda_module.get_global("triangles").set_value(ptr_triangles);

	delete [] aggregated_bvh_nodes;
	delete [] aggregated_triangles;

	ptr_sky_data = CUDAMemory::malloc(scene.sky.data, scene.sky.width * scene.sky.height);

	cuda_module.get_global("sky_width") .set_value(scene.sky.width);
	cuda_module.get_global("sky_height").set_value(scene.sky.height);
	cuda_module.get_global("sky_data")  .set_value(ptr_sky_data);

	for (int i = 1; i < PMJ_NUM_SEQUENCES; i++) {
		PMJ::shuffle(i);
	}

	ptr_pmj_samples = CUDAMemory::malloc<PMJ::Point>(PMJ_NUM_SEQUENCES * PMJ_NUM_SAMPLES_PER_SEQUENCE);
	CUDAMemory::memcpy(ptr_pmj_samples, PMJ::samples, PMJ_NUM_SEQUENCES * PMJ_NUM_SAMPLES_PER_SEQUENCE);
	cuda_module.get_global("pmj_samples").set_value(ptr_pmj_samples);

	ptr_blue_noise_textures = CUDAMemory::malloc<unsigned short>(BLUE_NOISE_NUM_TEXTURES * BLUE_NOISE_TEXTURE_DIM * BLUE_NOISE_TEXTURE_DIM);
	CUDAMemory::memcpy(ptr_blue_noise_textures, reinterpret_cast<unsigned short *>(BlueNoise::textures), BLUE_NOISE_NUM_TEXTURES * BLUE_NOISE_TEXTURE_DIM * BLUE_NOISE_TEXTURE_DIM);
	cuda_module.get_global("blue_noise_textures").set_value(ptr_blue_noise_textures);

	ray_buffer_trace.init(BATCH_SIZE);
	cuda_module.get_global("ray_buffer_trace").set_value(ray_buffer_trace);

	global_ray_buffer_shade_diffuse               = cuda_module.get_global("ray_buffer_shade_diffuse");
	global_ray_buffer_shade_dielectric_and_glossy = cuda_module.get_global("ray_buffer_shade_dielectric_and_glossy");
	global_ray_buffer_shadow                      = cuda_module.get_global("ray_buffer_shadow");

	global_camera      = cuda_module.get_global("camera");
	global_config      = cuda_module.get_global("config");
	global_svgf_data   = cuda_module.get_global("svgf_data");
	global_pixel_query = cuda_module.get_global("pixel_query");

	global_lights_total_weight = cuda_module.get_global("lights_total_weight");
	global_lights_total_weight.set_value(0.0f);

	// Initialize CUDA Events used for timing
	int display_order = 0;
	event_desc_primary = { display_order++, "Primary", "Primary" };

	for (int i = 0; i < MAX_BOUNCES; i++) {
		const int len = 16;
		char    * category = new char[len];
		sprintf_s(category, len, "Bounce %i", i);

		event_desc_trace           [i] = { display_order, category, "Trace" };
		event_desc_sort            [i] = { display_order, category, "Sort" };
		event_desc_shade_diffuse   [i] = { display_order, category, "Diffuse" };
		event_desc_shade_dielectric[i] = { display_order, category, "Dielectric" };
		event_desc_shade_glossy    [i] = { display_order, category, "Glossy" };
		event_desc_shadow_trace    [i] = { display_order, category, "Shadow" };

		display_order++;
	}

	event_desc_svgf_reproject = { display_order, "SVGF", "Reproject" };
	event_desc_svgf_variance  = { display_order, "SVGF", "Variance" };

	for (int i = 0; i < MAX_ATROUS_ITERATIONS; i++) {
		const int len = 16;
		char    * name = new char[len];
		sprintf_s(name, len, "A Trous %i", i);

		event_desc_svgf_atrous[i] = { display_order, "SVGF", name };
	}
	event_desc_svgf_finalize = { display_order++, "SVGF", "Finalize" };

	event_desc_taa         = { display_order, "Post", "TAA" };
	event_desc_reconstruct = { display_order, "Post", "Reconstruct" };
	event_desc_accumulate  = { display_order, "Post", "Accumulate" };

	event_desc_end = { ++display_order, "END", "END" };

	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH:  tlas.nodes_2 = CUDAMemory::malloc_pinned<BVHNode2>(2 * scene.meshes.size()); break;
		case BVHType::QBVH:  tlas.nodes_4 = CUDAMemory::malloc_pinned<BVHNode4>(2 * scene.meshes.size()); break;
		case BVHType::CWBVH: tlas.nodes_8 = CUDAMemory::malloc_pinned<BVHNode8>(2 * scene.meshes.size()); break;
		default: abort();
	}

	scene.camera.update(0.0f);
	scene.update(0.0f);

	scene.has_diffuse    = false;
	scene.has_dielectric = false;
	scene.has_glossy     = false;
	scene.has_lights     = false;

	invalidated_scene     = true;
	invalidated_materials = true;
	invalidated_config    = true;

	unsigned long long bytes_available = CUDAContext::get_available_memory();
	unsigned long long bytes_allocated = CUDAContext::total_memory - bytes_available;

	printf("CUDA Memory allocated: %8llu KB (%6llu MB)\n",   bytes_allocated >> 10, bytes_allocated >> 20);
	printf("CUDA Memory free:      %8llu KB (%6llu MB)\n\n", bytes_available >> 10, bytes_available >> 20);
}

void Pathtracer::cuda_free() {
	CUDAMemory::free(ptr_materials);

	if (scene.asset_manager.textures.size() > 0) {
		CUDAMemory::free(ptr_textures);

		for (int i = 0; i < scene.asset_manager.textures.size(); i++) {
			CUDAMemory::free_array(texture_arrays[i]);
			CUDAMemory::free_texture(textures[i].texture);
		}

		delete [] textures;
		delete [] texture_arrays;
	}

	CUDAMemory::free_pinned(pinned_buffer_sizes);
	CUDAMemory::free_pinned(pinned_mesh_bvh_root_indices);
	CUDAMemory::free_pinned(pinned_mesh_material_ids);
	CUDAMemory::free_pinned(pinned_mesh_transforms);
	CUDAMemory::free_pinned(pinned_mesh_transforms_inv);
	CUDAMemory::free_pinned(pinned_mesh_transforms_prev);
	CUDAMemory::free_pinned(pinned_light_mesh_prob_alias);
	CUDAMemory::free_pinned(pinned_light_mesh_transform_index);

	delete [] light_mesh_probabilites;

	CUDAMemory::free(ptr_mesh_bvh_root_indices);
	CUDAMemory::free(ptr_mesh_material_ids);
	CUDAMemory::free(ptr_mesh_transforms);
	CUDAMemory::free(ptr_mesh_transforms_inv);
	CUDAMemory::free(ptr_mesh_transforms_prev);

	CUDAMemory::free(ptr_bvh_nodes);

	CUDAMemory::free(ptr_triangles);

	CUDAMemory::free(ptr_sky_data);

	CUDAMemory::free(ptr_pmj_samples);

	if (scene.has_lights) {
		CUDAMemory::free(ptr_light_indices);
		CUDAMemory::free(ptr_light_prob_alias);

		CUDAMemory::free(ptr_light_mesh_prob_alias);
		CUDAMemory::free(ptr_light_mesh_first_index_and_triangle_count);
		CUDAMemory::free(ptr_light_mesh_transform_index);
	}

                                                  ray_buffer_trace                      .free();
	if (scene.has_diffuse)                        ray_buffer_shade_diffuse              .free();
	if (scene.has_dielectric || scene.has_glossy) ray_buffer_shade_dielectric_and_glossy.free();
	if (scene.has_lights)                         ray_buffer_shadow                     .free();

	tlas_bvh_builder.free();
	if (config.bvh_type == BVHType::CWBVH) {
		tlas_converter_cwbvh.free();
	}

	delete [] reverse_indices;

	delete [] mesh_data_bvh_offsets;
	delete [] mesh_data_triangle_offsets;

	resize_free();

	cuda_module.free();
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
	ptr_frame_buffer_albedo = CUDAMemory::malloc<float4>(screen_pitch * height);
	cuda_module.get_global("frame_buffer_albedo")  .set_value(ptr_frame_buffer_albedo);

	ptr_frame_buffer_direct   = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_frame_buffer_indirect = CUDAMemory::malloc<float4>(screen_pitch * height);
	cuda_module.get_global("frame_buffer_direct")  .set_value(ptr_frame_buffer_direct);
	cuda_module.get_global("frame_buffer_indirect").set_value(ptr_frame_buffer_indirect);

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

	kernel_generate        .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_generate        .block_dim_x), 1, 1);
	kernel_sort            .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_sort            .block_dim_x), 1, 1);
	kernel_shade_diffuse   .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_shade_diffuse   .block_dim_x), 1, 1);
	kernel_shade_dielectric.set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_shade_dielectric.block_dim_x), 1, 1);
	kernel_shade_glossy    .set_grid_dim(Math::divide_round_up(BATCH_SIZE, kernel_shade_glossy    .block_dim_x), 1, 1);

	scene.camera.resize(width, height);
	invalidated_camera = true;

	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(Math::min(BATCH_SIZE, pixel_count));
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	frames_accumulated = 0;

	if (config.enable_svgf) svgf_init();
}

void Pathtracer::resize_free() {
	CUDACALL(cuStreamSynchronize(stream_memset));

	CUDAMemory::free(ptr_frame_buffer_albedo);

	CUDAMemory::resource_unregister(resource_accumulator);
	CUDAMemory::free_surface(surf_accumulator);

	CUDAMemory::free(ptr_frame_buffer_direct);
	CUDAMemory::free(ptr_frame_buffer_indirect);

	if (config.enable_svgf) svgf_free();
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
	ptr_frame_buffer_moment = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	cuda_module.get_global("frame_buffer_moment").set_value(ptr_frame_buffer_moment);

	ptr_frame_buffer_direct_alt   = CUDAMemory::malloc<float4>(screen_pitch * screen_height);
	ptr_frame_buffer_indirect_alt = CUDAMemory::malloc<float4>(screen_pitch * screen_height);

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

	CUDAMemory::free(ptr_frame_buffer_moment);

	CUDAMemory::free(ptr_frame_buffer_direct_alt);
	CUDAMemory::free(ptr_frame_buffer_indirect_alt);

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
		int first_triangle_index;
		int triangle_count;

		double total_area;
	};
	Array<LightMeshData> light_mesh_datas;

	using It = decltype(mesh_data_used_as_lights)::Iterator;

	for (auto it = mesh_data_used_as_lights.begin(); it != mesh_data_used_as_lights.end(); ++it) {
		MeshDataHandle  mesh_data_handle = MeshDataHandle { it.get_key() };
		Array<Mesh *> & meshes           = it.get_value();

		const MeshData & mesh_data = scene.asset_manager.get_mesh_data(mesh_data_handle);

		int light_index = light_mesh_datas.size();

		LightMeshData & light_mesh_data = light_mesh_datas.emplace_back();
		light_mesh_data.first_triangle_index = light_triangles.size();
		light_mesh_data.triangle_count = mesh_data.triangle_count;
		light_mesh_data.total_area = 0.0f;

		for (int t = 0; t < mesh_data.triangle_count; t++) {
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

			mesh->light.weight               = power * light_mesh_data.total_area;
			mesh->light.first_triangle_index = light_mesh_data.first_triangle_index;
			mesh->light.triangle_count       = light_mesh_data.triangle_count;
		}
	}

	if (light_triangles.size() > 0) {
		int       * light_indices       = new int      [light_triangles.size()];
		double    * light_probabilities = new double   [light_triangles.size()];
		ProbAlias * light_prob_alias    = new ProbAlias[light_triangles.size()];

		for (int m = 0; m < light_mesh_datas.size(); m++) {
			const LightMeshData & light_mesh_data = light_mesh_datas[m];

			for (int i = light_mesh_data.first_triangle_index; i < light_mesh_data.first_triangle_index + light_mesh_data.triangle_count; i++) {
				light_indices      [i] = light_triangles[i].index;
				light_probabilities[i] = light_triangles[i].area / light_mesh_data.total_area;
			}

			Util::init_alias_method(
				light_mesh_data.triangle_count,
				light_probabilities + light_mesh_data.first_triangle_index,
				light_prob_alias    + light_mesh_data.first_triangle_index
			);
		}

		ptr_light_indices    = CUDAMemory::malloc(light_indices,    light_triangles.size());
		ptr_light_prob_alias = CUDAMemory::malloc(light_prob_alias, light_triangles.size());

		cuda_module.get_global("light_indices")   .set_value(ptr_light_indices);
		cuda_module.get_global("light_prob_alias").set_value(ptr_light_prob_alias);

		delete [] light_indices;
		delete [] light_probabilities;
		delete [] light_prob_alias;

		cuda_module.get_global("light_mesh_count").set_value(light_mesh_count);

		if (ptr_light_mesh_prob_alias                    .ptr != NULL) CUDAMemory::free(ptr_light_mesh_prob_alias);
		if (ptr_light_mesh_first_index_and_triangle_count.ptr != NULL) CUDAMemory::free(ptr_light_mesh_first_index_and_triangle_count);
		if (ptr_light_mesh_transform_index               .ptr != NULL) CUDAMemory::free(ptr_light_mesh_transform_index);

		// The Device pointers below are only filled in and copied to the GPU once the TLAS is constructed,
		// therefore the scene_invalidated flag is required to be set.
		invalidated_scene = true;

		ptr_light_mesh_prob_alias                     = CUDAMemory::malloc<ProbAlias>(light_mesh_count);
		ptr_light_mesh_first_index_and_triangle_count = CUDAMemory::malloc<int2>     (light_mesh_count);
		ptr_light_mesh_transform_index                = CUDAMemory::malloc<int>      (light_mesh_count);

		cuda_module.get_global("light_mesh_prob_alias")                    .set_value(ptr_light_mesh_prob_alias);
		cuda_module.get_global("light_mesh_first_index_and_triangle_count").set_value(ptr_light_mesh_first_index_and_triangle_count);
		cuda_module.get_global("light_mesh_transform_index")               .set_value(ptr_light_mesh_transform_index);
	}
}

// Construct Top Level Acceleration Structure (TLAS) over the Meshes in the Scene
void Pathtracer::build_tlas() {
	tlas_bvh_builder.build(scene.meshes.data(), scene.meshes.size());

	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH:  tlas = tlas_raw; break;
		case BVHType::QBVH:  tlas_converter_qbvh .build(tlas_raw); break;
		case BVHType::CWBVH: tlas_converter_cwbvh.build(tlas_raw); break;
		default: abort();
	}

	assert(tlas.index_count == scene.meshes.size());
	assert(tlas.node_count <= 2 * scene.meshes.size());
	CUDAMemory::memcpy(ptr_bvh_nodes, tlas.nodes_raw, tlas.node_count * BVH::node_size());

	int    light_mesh_count    = 0;
	double lights_total_weight = 0.0;

	for (int i = 0; i < scene.meshes.size(); i++) {
		const Mesh & mesh = scene.meshes[tlas.indices[i]];

		pinned_mesh_bvh_root_indices[i] = mesh_data_bvh_offsets[mesh.mesh_data_handle.handle] | (mesh.has_identity_transform() << 31);

		assert(mesh.material_handle.handle != INVALID);
		pinned_mesh_material_ids[i] = mesh.material_handle.handle;

		memcpy(pinned_mesh_transforms     [i].cells, mesh.transform     .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_inv [i].cells, mesh.transform_inv .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_prev[i].cells, mesh.transform_prev.cells, sizeof(Matrix3x4));

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

	CUDAMemory::memcpy(ptr_mesh_bvh_root_indices,  pinned_mesh_bvh_root_indices, scene.meshes.size());
	CUDAMemory::memcpy(ptr_mesh_material_ids,      pinned_mesh_material_ids,     scene.meshes.size());
	CUDAMemory::memcpy(ptr_mesh_transforms,        pinned_mesh_transforms,       scene.meshes.size());
	CUDAMemory::memcpy(ptr_mesh_transforms_inv,    pinned_mesh_transforms_inv,   scene.meshes.size());
	CUDAMemory::memcpy(ptr_mesh_transforms_prev,   pinned_mesh_transforms_prev,  scene.meshes.size());

	if (light_mesh_count > 0) {
		for (int i = 0; i < light_mesh_count; i++) {
			light_mesh_probabilites[i] /= lights_total_weight;
		}
		Util::init_alias_method(light_mesh_count, light_mesh_probabilites, pinned_light_mesh_prob_alias);

		CUDAMemory::memcpy(ptr_light_mesh_prob_alias,                     pinned_light_mesh_prob_alias,                     light_mesh_count);
		CUDAMemory::memcpy(ptr_light_mesh_first_index_and_triangle_count, pinned_light_mesh_first_index_and_triangle_count, light_mesh_count);
		CUDAMemory::memcpy(ptr_light_mesh_transform_index,                pinned_light_mesh_transform_index,                light_mesh_count);
	}

	global_lights_total_weight.set_value(float(lights_total_weight));
}

void Pathtracer::update(float delta) {
	if (invalidated_config && config.enable_svgf && scene.camera.aperture_radius > 0.0f) {
		puts("WARNING: SVGF and DoF cannot simultaneously be enabled!");
		scene.camera.aperture_radius = 0.0f;
	}

	if (invalidated_materials) {
		const Array<Material> & materials = scene.asset_manager.materials;

		Material::Type * cuda_material_types = reinterpret_cast<Material::Type *>(new unsigned char[materials.size() * sizeof(Material::Type)]);
		CUDAMaterial   * cuda_materials      = reinterpret_cast<CUDAMaterial   *>(new unsigned char[materials.size() * sizeof(CUDAMaterial)]);

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
				case Material::Type::DIELECTRIC: {
					cuda_materials[i].dielectric.negative_absorption = Vector3( // Absorption = -log(Transmittance), so -A = log(T)
						logf(material.transmittance.x),
						logf(material.transmittance.y),
						logf(material.transmittance.z)
					);
					cuda_materials[i].dielectric.index_of_refraction = Math::max(material.index_of_refraction, 1.0001f);
					break;
				}
				case Material::Type::GLOSSY: {
					cuda_materials[i].glossy.diffuse    = material.diffuse;
					cuda_materials[i].glossy.texture_id = material.texture_id.handle;
					cuda_materials[i].glossy.eta        = Vector3::max(material.eta, Vector3(1.0001f));
					cuda_materials[i].glossy.k          = material.k;
					cuda_materials[i].glossy.roughness  = Math::max(material.linear_roughness * material.linear_roughness, 1e-6f);
					break;
				}
				default: abort();
			}
		}

		CUDAMemory::memcpy(ptr_material_types, cuda_material_types, materials.size());
		CUDAMemory::memcpy(ptr_materials,      cuda_materials,      materials.size());

		delete [] cuda_material_types;
		delete [] cuda_materials;

		bool had_diffuse    = scene.has_diffuse;
		bool had_dielectric = scene.has_dielectric;
		bool had_glossy     = scene.has_glossy;
		bool had_lights     = scene.has_lights;

		scene.check_materials();

		bool diffuse_changed              =  had_diffuse                   ^  scene.has_diffuse;
		bool dielectric_or_glossy_changed = (had_dielectric || had_glossy) ^ (scene.has_dielectric || scene.has_glossy);
		bool lights_changed               =  had_lights                    ^  scene.has_lights;

		// Handle (dis)appearance of Diffuse materials
		if (diffuse_changed) {
			if (scene.has_diffuse) {
				ray_buffer_shade_diffuse.init(BATCH_SIZE);
			} else {
				ray_buffer_shade_diffuse.free();
			}
			global_ray_buffer_shade_diffuse.set_value(ray_buffer_shade_diffuse);
		}

		// Handle (dis)appearance of Dielectric OR Glossy materials (they share the same Material buffer)
		if (dielectric_or_glossy_changed) {
			if (scene.has_dielectric || scene.has_glossy) {
				ray_buffer_shade_dielectric_and_glossy.init(BATCH_SIZE);
			} else {
				ray_buffer_shade_dielectric_and_glossy.free();
			}
			global_ray_buffer_shade_dielectric_and_glossy.set_value(ray_buffer_shade_dielectric_and_glossy);
		}

		// Handle (dis)appearance of Light materials
		if (lights_changed) {
			if (scene.has_lights) {
				ray_buffer_shadow.init(BATCH_SIZE);

				invalidated_scene = true;
			} else {
				ray_buffer_shadow.free();

				CUDAMemory::free(ptr_light_mesh_prob_alias);
				CUDAMemory::free(ptr_light_mesh_first_index_and_triangle_count);
				CUDAMemory::free(ptr_light_mesh_transform_index);

				global_lights_total_weight.set_value(0.0f);

				for (int i = 0; i < scene.meshes.size(); i++) {
					scene.meshes[i].light.weight = 0.0f;
				}
			}

			global_ray_buffer_shadow.set_value(ray_buffer_shadow);
		}

		if (scene.has_lights) calc_light_power();

		frames_accumulated = 0;
		invalidated_materials = false;
	}

	if (config.enable_scene_update) {
		scene.update(delta);
		invalidated_scene = true;
	} else if (config.enable_svgf || invalidated_scene) {
		scene.camera.update(0.0f);
		scene.update(0.0f);
	}

	if (invalidated_scene) {
		build_tlas();

		// If SVGF is enabled we can handle Scene updates using reprojection,
		// otherwise 'frames_accumulated' needs to be reset in order to avoid ghosting
		if (!config.enable_svgf) {
			frames_accumulated = 0;
		}

		invalidated_scene = false;
	}

	scene.camera.update(delta);

	if (scene.camera.moved || invalidated_camera) {
		// Upload Camera
		struct CUDACamera {
			Vector3 position;
			Vector3 bottom_left_corner;
			Vector3 x_axis;
			Vector3 y_axis;
			float pixel_spread_angle;
			float aperture_radius;
			float focal_distance;
		} cuda_camera;

		cuda_camera.position           = scene.camera.position;
		cuda_camera.bottom_left_corner = scene.camera.bottom_left_corner_rotated;
		cuda_camera.x_axis             = scene.camera.x_axis_rotated;
		cuda_camera.y_axis             = scene.camera.y_axis_rotated;
		cuda_camera.pixel_spread_angle = scene.camera.pixel_spread_angle;
		cuda_camera.aperture_radius    = scene.camera.aperture_radius;
		cuda_camera.focal_distance     = scene.camera.focal_distance;

		global_camera.set_value(cuda_camera);

		if (!config.enable_svgf) {
			frames_accumulated = 0;
		}

		invalidated_camera = false;
	}

	if (pixel_query_status == PixelQueryStatus::OUTPUT_READY) {
		CUDAMemory::memcpy(&pixel_query, CUDAMemory::Ptr<PixelQuery>(global_pixel_query.ptr));

		if (pixel_query.mesh_id != INVALID) {
			pixel_query.mesh_id = tlas.indices[pixel_query.mesh_id];
		}

		// Reset pixel query
		pixel_query.pixel_index = INVALID;
		CUDAMemory::memcpy(CUDAMemory::Ptr<PixelQuery>(global_pixel_query.ptr), &pixel_query);

		pixel_query_status = PixelQueryStatus::INACTIVE;
	}

	if (config.enable_svgf) {
		struct SVGFData {
			alignas(16) Matrix4 view_projection;
			alignas(16) Matrix4 view_projection_prev;
		} svgf_data;

		svgf_data.view_projection      = scene.camera.view_projection;
		svgf_data.view_projection_prev = scene.camera.view_projection_prev;

		global_svgf_data.set_value(svgf_data);
	}

	if (invalidated_config) {
		frames_accumulated = 0;

		global_config.set_value(config);
	} else if (config.enable_svgf) {
		frames_accumulated = (frames_accumulated + 1) & 255;
	} else if (scene.camera.moved) {
		frames_accumulated = 0;
	} else {
		frames_accumulated++;
	}
}

void Pathtracer::render() {
	event_pool.reset();

	CUDACALL(cuStreamSynchronize(stream_memset));

	int pixels_left = pixel_count;
	int batch_size  = Math::min(BATCH_SIZE, pixel_count);

	// Render in batches of BATCH_SIZE pixels at a time
	while (pixels_left > 0) {
		int pixel_offset = pixel_count - pixels_left;
		int pixel_count  = Math::min(batch_size, pixels_left);

		event_pool.record(event_desc_primary);

		// Generate primary Rays from the current Camera orientation
		kernel_generate.execute(frames_accumulated, pixel_offset, pixel_count);

		for (int bounce = 0; bounce < config.num_bounces; bounce++) {
			// Extend all Rays that are still alive to their next Triangle intersection
			event_pool.record(event_desc_trace[bounce]);
			kernel_trace->execute(bounce);

			event_pool.record(event_desc_sort[bounce]);
			kernel_sort.execute(bounce, frames_accumulated);

			// Process the various Material types in different Kernels
			if (scene.has_diffuse) {
				event_pool.record(event_desc_shade_diffuse[bounce]);
				kernel_shade_diffuse.execute(bounce, frames_accumulated);
			}

			if (scene.has_dielectric) {
				event_pool.record(event_desc_shade_dielectric[bounce]);
				kernel_shade_dielectric.execute(bounce, frames_accumulated);
			}

			if (scene.has_glossy) {
				event_pool.record(event_desc_shade_glossy[bounce]);
				kernel_shade_glossy.execute(bounce, frames_accumulated);
			}

			// Trace shadow Rays
			if (scene.has_lights && config.enable_next_event_estimation) {
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

	if (config.enable_svgf) {
		// Temporal reprojection + integration
		event_pool.record(event_desc_svgf_reproject);
		kernel_svgf_reproject.execute(frames_accumulated);

		CUdeviceptr direct_in    = ptr_frame_buffer_direct    .ptr;
		CUdeviceptr direct_out   = ptr_frame_buffer_direct_alt.ptr;
		CUdeviceptr indirect_in  = ptr_frame_buffer_indirect    .ptr;
		CUdeviceptr indirect_out = ptr_frame_buffer_indirect_alt.ptr;

		if (config.enable_spatial_variance) {
			// Estimate Variance spatially
			event_pool.record(event_desc_svgf_variance);
			kernel_svgf_variance.execute(direct_in, indirect_in, direct_out, indirect_out);

			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);
		}

		// À-Trous Filter
		for (int i = 0; i < config.num_atrous_iterations; i++) {
			int step_size = 1 << i;

			event_pool.record(event_desc_svgf_atrous[i]);
			kernel_svgf_atrous.execute(direct_in, indirect_in, direct_out, indirect_out, step_size);

			// Ping-Pong the Frame Buffers
			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);
		}

		event_pool.record(event_desc_svgf_finalize);
		kernel_svgf_finalize.execute(direct_in, indirect_in);

		if (config.enable_taa) {
			event_pool.record(event_desc_taa);

			kernel_taa         .execute(frames_accumulated);
			kernel_taa_finalize.execute();
		}
	} else {
		event_pool.record(event_desc_accumulate);
		kernel_accumulate.execute(float(frames_accumulated));
	}

	event_pool.record(event_desc_end);

	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(batch_size);
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	if (config.enable_albedo) CUDAMemory::memset_async(ptr_frame_buffer_albedo, 0, screen_pitch * screen_height, stream_memset);
	CUDAMemory::memset_async(ptr_frame_buffer_direct,   0, screen_pitch * screen_height, stream_memset);
	CUDAMemory::memset_async(ptr_frame_buffer_indirect, 0, screen_pitch * screen_height, stream_memset);

	// If a pixel query was previously pending, it has just been resolved in the current frame
	if (pixel_query_status == PixelQueryStatus::PENDING) {
		pixel_query_status =  PixelQueryStatus::OUTPUT_READY;
	}
}

void Pathtracer::set_pixel_query(int x, int y) {
	if (x < 0 || y < 0 || x >= screen_width || y >= screen_height) return;

	y = screen_height - y; // Y-coordinate is inverted

	pixel_query.pixel_index = x + y * screen_pitch;
	pixel_query.mesh_id     = INVALID;
	pixel_query.triangle_id = INVALID;
	pixel_query.material_id = INVALID;
	global_pixel_query.set_value(pixel_query);

	pixel_query_status = PixelQueryStatus::PENDING;
}
