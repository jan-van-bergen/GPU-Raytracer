#include "Pathtracer.h"

#include <algorithm>

#include <GL/glew.h>

#include "CUDA/CUDAContext.h"

#include "Assets/MeshData.h"
#include "Assets/Material.h"
#include "Assets/MitsubaLoader.h"

#include "Math/Vector4.h"

#include "Util/Random.h"
#include "Util/BlueNoise.h"

#include "Util/Util.h"
#include "Util/ScopeTimer.h"

void Pathtracer::init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle) {
	scene.init(scene_name, sky_name);

	cuda_init(frame_buffer_handle, SCREEN_WIDTH, SCREEN_HEIGHT);

	CUDACALL(cuStreamCreate(&stream_memset, CU_STREAM_NON_BLOCKING));
}

void Pathtracer::cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) {
	// Init CUDA Module and its Kernels
	cuda_module.init("CUDA_Source/Pathtracer.cu", CUDAContext::compute_capability, MAX_REGISTERS);
	
	kernel_generate        .init(&cuda_module, "kernel_generate");
	kernel_trace           .init(&cuda_module, "kernel_trace");
	kernel_sort            .init(&cuda_module, "kernel_sort");
	kernel_shade_diffuse   .init(&cuda_module, "kernel_shade_diffuse");
	kernel_shade_dielectric.init(&cuda_module, "kernel_shade_dielectric");
	kernel_shade_glossy    .init(&cuda_module, "kernel_shade_glossy");
	kernel_trace_shadow    .init(&cuda_module, "kernel_trace_shadow");
	kernel_svgf_reproject  .init(&cuda_module, "kernel_svgf_reproject");
	kernel_svgf_variance   .init(&cuda_module, "kernel_svgf_variance");
	kernel_svgf_atrous     .init(&cuda_module, "kernel_svgf_atrous");
	kernel_svgf_finalize   .init(&cuda_module, "kernel_svgf_finalize");
	kernel_taa             .init(&cuda_module, "kernel_taa");
	kernel_taa_finalize    .init(&cuda_module, "kernel_taa_finalize");
	kernel_accumulate      .init(&cuda_module, "kernel_accumulate");

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
	
#if BVH_TYPE == BVH_CWBVH
	static constexpr int BVH_STACK_ELEMENT_SIZE = 8; // CWBVH uses a stack of int2's (8 bytes)
#else
	static constexpr int BVH_STACK_ELEMENT_SIZE = 4; // Other BVH's use a stack of ints (4 bytes)
#endif

	CUoccupancyB2DSize block_size_to_shared_memory = [](int block_size) {
		return size_t(block_size) * SHARED_STACK_SIZE * BVH_STACK_ELEMENT_SIZE;
	};

	int grid, block;
	CUDACALL(cuOccupancyMaxPotentialBlockSize(&grid, &block, kernel_trace.kernel, block_size_to_shared_memory, 0, 0)); 
	
	int block_x = WARP_SIZE;
	int block_y = block / WARP_SIZE;

	kernel_trace       .set_block_dim(block_x, block_y, 1);
	kernel_trace_shadow.set_block_dim(block_x, block_y, 1);

	kernel_trace       .set_grid_dim(1, grid, 1);
	kernel_trace_shadow.set_grid_dim(1, grid, 1);
	
	kernel_trace       .set_shared_memory(block_size_to_shared_memory(block));
	kernel_trace_shadow.set_shared_memory(block_size_to_shared_memory(block));

	printf("\nConfiguration picked for Tracing kernels:\n    Block Size: %i x %i\n    Grid Size:  %i\n\n", block_x, block_y, grid);
	
	pinned_buffer_sizes = CUDAMemory::malloc_pinned<BufferSizes>();
	pinned_buffer_sizes->reset(batch_size);
	
	global_buffer_sizes = cuda_module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	resize_init(frame_buffer_handle, screen_width, screen_height);
	
	// Set global Material table
	ptr_material_types = CUDAMemory::malloc<Material::Type>(scene.asset_manager.materials.size());
	ptr_materials      = CUDAMemory::malloc<CUDAMaterial>  (scene.asset_manager.materials.size());
	cuda_module.get_global("material_types").set_value(ptr_material_types);
	cuda_module.get_global("materials")     .set_value(ptr_materials);
	
	scene.asset_manager.wait_until_textures_loaded();

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
				int level_height         = texture.height >> level;
				
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

	BVHNodeType * aggregated_bvh_nodes = new BVHNodeType[aggregated_bvh_node_count];
	Triangle    * aggregated_triangles = new Triangle   [aggregated_triangle_count];
	int         * aggregated_indices   = new int        [aggregated_index_count];

	for (int m = 0; m < mesh_data_count; m++) {
		const MeshData & mesh_data = scene.asset_manager.mesh_datas[m];

		for (int n = 0; n < mesh_data.bvh.node_count; n++) {
			BVHNodeType & node = aggregated_bvh_nodes[mesh_data_bvh_offsets[m] + n];

			node = mesh_data.bvh.nodes[n];

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
			if (node.is_leaf()) {
				node.first += mesh_data_index_offsets[m];
			} else {
				node.left += mesh_data_bvh_offsets[m];
			}
#elif BVH_TYPE == BVH_QBVH
			int child_count = node.get_child_count();
			for (int c = 0; c < child_count; c++) {
				if (node.is_leaf(c)) {
					node.get_index(c) += mesh_data_index_offsets[m];
				} else {
					node.get_index(c) += mesh_data_bvh_offsets[m];
				}
			}
#elif BVH_TYPE == BVH_CWBVH
			node.base_index_child    += mesh_data_bvh_offsets[m];
			node.base_index_triangle += mesh_data_index_offsets[m];
#endif
		}

		for (int t = 0; t < mesh_data.triangle_count; t++) {
			aggregated_triangles[mesh_data_triangle_offsets[m] + t] = mesh_data.triangles[t];
		}

		for (int i = 0; i < mesh_data.bvh.index_count; i++) {
			aggregated_indices[mesh_data_index_offsets[m] + i] = mesh_data.bvh.indices[i] + mesh_data_triangle_offsets[m];
		}
	}

	FREEA(mesh_data_index_offsets);

	pinned_mesh_bvh_root_indices        = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());
	pinned_mesh_material_ids            = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());
	pinned_mesh_transforms              = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_mesh_transforms_inv          = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_mesh_transforms_prev         = CUDAMemory::malloc_pinned<Matrix3x4>(scene.meshes.size());
	pinned_light_mesh_transform_indices = CUDAMemory::malloc_pinned<int>      (scene.meshes.size());
	pinned_light_mesh_area_scaled       = CUDAMemory::malloc_pinned<float>    (scene.meshes.size());

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
	
	ptr_bvh_nodes = CUDAMemory::malloc<BVHNodeType>(aggregated_bvh_nodes, aggregated_bvh_node_count);

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	cuda_module.get_global("bvh_nodes").set_value(ptr_bvh_nodes);
#elif BVH_TYPE == BVH_QBVH
	cuda_module.get_global("qbvh_nodes").set_value(ptr_bvh_nodes);
#elif BVH_TYPE == BVH_CWBVH
	cuda_module.get_global("cwbvh_nodes").set_value(ptr_bvh_nodes);
#endif

	tlas_bvh_builder.init(&tlas_raw, scene.meshes.size(), 1);

	tlas_raw.node_count = scene.meshes.size() * 2;
#if BVH_TYPE == BVH_QBVH || BVH_TYPE == BVH_CWBVH
	tlas_converter.init(&tlas, tlas_raw);
#endif

	CUDATriangle * triangles = new CUDATriangle[aggregated_index_count];
	
	reverse_indices = new int[aggregated_index_count];

	for (int i = 0; i < aggregated_index_count; i++) {
		int index = aggregated_indices[i];

		assert(index < aggregated_triangle_count);

		triangles[i].position_0      = aggregated_triangles[index].position_0;
		triangles[i].position_edge_1 = aggregated_triangles[index].position_1 - aggregated_triangles[index].position_0;
		triangles[i].position_edge_2 = aggregated_triangles[index].position_2 - aggregated_triangles[index].position_0;

		triangles[i].normal_0      = aggregated_triangles[index].normal_0;
		triangles[i].normal_edge_1 = aggregated_triangles[index].normal_1 - aggregated_triangles[index].normal_0;
		triangles[i].normal_edge_2 = aggregated_triangles[index].normal_2 - aggregated_triangles[index].normal_0;

		triangles[i].tex_coord_0      = aggregated_triangles[index].tex_coord_0;
		triangles[i].tex_coord_edge_1 = aggregated_triangles[index].tex_coord_1 - aggregated_triangles[index].tex_coord_0;
		triangles[i].tex_coord_edge_2 = aggregated_triangles[index].tex_coord_2 - aggregated_triangles[index].tex_coord_0;

		reverse_indices[index] = i;
	}

	ptr_triangles = CUDAMemory::malloc(triangles, aggregated_index_count);
	
	cuda_module.get_global("triangles").set_value(ptr_triangles);
	
	delete [] aggregated_bvh_nodes;
	delete [] aggregated_indices;
	delete [] aggregated_triangles;

	delete [] triangles;
	
	ptr_sky_data = CUDAMemory::malloc(scene.sky.data, scene.sky.width * scene.sky.height);

	cuda_module.get_global("sky_width") .set_value(scene.sky.width);
	cuda_module.get_global("sky_height").set_value(scene.sky.height);
	cuda_module.get_global("sky_data")  .set_value(ptr_sky_data);
	
	// Set Blue Noise Sampler globals
	ptr_sobol_256spp_256d = CUDAMemory::malloc(sobol_256spp_256d);
	ptr_scrambling_tile   = CUDAMemory::malloc(scrambling_tile);
	ptr_ranking_tile      = CUDAMemory::malloc(ranking_tile);

	cuda_module.get_global("sobol_256spp_256d").set_value(ptr_sobol_256spp_256d);
	cuda_module.get_global("scrambling_tile")  .set_value(ptr_scrambling_tile);
	cuda_module.get_global("ranking_tile")     .set_value(ptr_ranking_tile);
	
	ray_buffer_trace.init(batch_size);
	cuda_module.get_global("ray_buffer_trace").set_value(ray_buffer_trace);
	
	global_ray_buffer_shade_diffuse               = cuda_module.get_global("ray_buffer_shade_diffuse");
	global_ray_buffer_shade_dielectric_and_glossy = cuda_module.get_global("ray_buffer_shade_dielectric_and_glossy");
	global_ray_buffer_shadow                      = cuda_module.get_global("ray_buffer_shadow");

	global_camera      = cuda_module.get_global("camera");
	global_settings    = cuda_module.get_global("settings");
	global_svgf_data   = cuda_module.get_global("svgf_data");
	global_pixel_query = cuda_module.get_global("pixel_query");
	
	global_lights_total_power = cuda_module.get_global("lights_total_power");
	global_lights_total_power.set_value(0.0f);

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
	
	// Realloc as pinned memory
	delete [] tlas.nodes;
	tlas.nodes = CUDAMemory::malloc_pinned<BVHNodeType>(2 * scene.meshes.size());
	
	scene.camera.update(0.0f, settings);
	scene.update(0.0f);

	scene.has_diffuse    = false;
	scene.has_dielectric = false;
	scene.has_glossy     = false;
	scene.has_lights     = false;

	invalidated_scene     = true;
	invalidated_materials = true;
	invalidated_settings  = true;

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
	CUDAMemory::free_pinned(pinned_light_mesh_transform_indices);
	CUDAMemory::free_pinned(pinned_light_mesh_area_scaled);

	CUDAMemory::free(ptr_mesh_bvh_root_indices);
	CUDAMemory::free(ptr_mesh_material_ids);
	CUDAMemory::free(ptr_mesh_transforms);
	CUDAMemory::free(ptr_mesh_transforms_inv);
	CUDAMemory::free(ptr_mesh_transforms_prev);

	CUDAMemory::free(ptr_bvh_nodes);

	CUDAMemory::free(ptr_triangles);
	
	CUDAMemory::free(ptr_sky_data);

	CUDAMemory::free(ptr_sobol_256spp_256d);
	CUDAMemory::free(ptr_scrambling_tile);
	CUDAMemory::free(ptr_ranking_tile);

	if (scene.has_lights) {
		CUDAMemory::free(ptr_light_indices);
		CUDAMemory::free(ptr_light_areas_cumulative);
	
		CUDAMemory::free(ptr_light_mesh_power_unscaled);
		CUDAMemory::free(ptr_light_mesh_triangle_count);
		CUDAMemory::free(ptr_light_mesh_triangle_first_index);

		CUDAMemory::free(ptr_light_mesh_power_scaled);
		CUDAMemory::free(ptr_light_mesh_transform_indices);
	}

                                                  ray_buffer_trace                      .free();
	if (scene.has_diffuse)                        ray_buffer_shade_diffuse              .free();
	if (scene.has_dielectric || scene.has_glossy) ray_buffer_shade_dielectric_and_glossy.free();
	if (scene.has_lights)                         ray_buffer_shadow                     .free();
	
	tlas_bvh_builder.free();
#if BVH_TYPE == BVH_CWBVH
	tlas_converter.free();
#endif

	delete [] reverse_indices;

	delete [] mesh_data_bvh_offsets;
	delete [] mesh_data_triangle_offsets;

	resize_free();

	cuda_module.free();
}

void Pathtracer::resize_init(unsigned frame_buffer_handle, int width, int height) {
	screen_width  = width;
	screen_height = height;
	screen_pitch  = Math::divide_round_up(width, WARP_SIZE) * WARP_SIZE;

	pixel_count = width * height;
	batch_size  = Math::min(BATCH_SIZE, pixel_count);

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

	kernel_generate        .set_grid_dim(Math::divide_round_up(batch_size, kernel_generate        .block_dim_x), 1, 1);
	kernel_sort            .set_grid_dim(Math::divide_round_up(batch_size, kernel_sort            .block_dim_x), 1, 1);
	kernel_shade_diffuse   .set_grid_dim(Math::divide_round_up(batch_size, kernel_shade_diffuse   .block_dim_x), 1, 1);
	kernel_shade_dielectric.set_grid_dim(Math::divide_round_up(batch_size, kernel_shade_dielectric.block_dim_x), 1, 1);
	kernel_shade_glossy    .set_grid_dim(Math::divide_round_up(batch_size, kernel_shade_glossy    .block_dim_x), 1, 1);
	
	scene.camera.resize(width, height);
	invalidated_camera = true;

	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(batch_size);
	global_buffer_sizes.set_value(*pinned_buffer_sizes);
	
	frames_accumulated = 0;

	if (settings.enable_svgf) svgf_init();
}

void Pathtracer::resize_free() {
	CUDACALL(cuStreamSynchronize(stream_memset));

	CUDAMemory::free(ptr_frame_buffer_albedo);

	CUDAMemory::resource_unregister(resource_accumulator);
	CUDAMemory::free_surface(surf_accumulator);

	CUDAMemory::free(ptr_frame_buffer_direct);
	CUDAMemory::free(ptr_frame_buffer_indirect);

	if (settings.enable_svgf) svgf_free();
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
	struct LightTriangle {
		int   index;
		float power;
	};
	std::vector<LightTriangle> light_triangles;

	struct LightMesh {
		int triangle_first_index;
		int triangle_count;

		float power;
	};
	std::vector<LightMesh> light_meshes;

	int mesh_data_count = scene.asset_manager.mesh_datas.size();
	
	int * light_mesh_data_indices = MALLOCA(int, mesh_data_count);
	memset(light_mesh_data_indices, -1, mesh_data_count * sizeof(int));

	// For every Mesh, check whether it is a Light based on its Material
	for (int m = 0; m < scene.meshes.size(); m++) {
		const Mesh     & mesh      = scene.meshes[m];
		const Material & material  = scene.asset_manager.get_material (mesh.material_handle);
		const MeshData & mesh_data = scene.asset_manager.get_mesh_data(mesh.mesh_data_handle);

		if (material.type == Material::Type::LIGHT) {
			light_mesh_data_indices[m] = light_meshes.size();

			LightMesh & light_mesh = light_meshes.emplace_back();
			light_mesh.triangle_first_index = light_triangles.size();
			light_mesh.triangle_count = mesh_data.triangle_count;

			for (int t = 0; t < mesh_data.triangle_count; t++) {
				const Triangle & triangle = mesh_data.triangles[t];
			
				float area = 0.5f * Vector3::length(Vector3::cross(
					triangle.position_1 - triangle.position_0,
					triangle.position_2 - triangle.position_0
				));
				float power = material.emission.x + material.emission.y + material.emission.z;

				light_triangles.push_back({ reverse_indices[mesh_data_triangle_offsets[m] + t], power * area });
			}
				
			// Sort Lights by power within each Mesh
			LightTriangle * triangles_begin = light_triangles.data() + light_mesh.triangle_first_index;
			LightTriangle * triangles_end   = triangles_begin        + light_mesh.triangle_count;

			assert(triangles_end > triangles_begin);
			std::sort(triangles_begin, triangles_end, [](const LightTriangle & a, const LightTriangle & b) { return a.power < b.power; });
		}
	}

	int   * light_indices          = new int  [light_triangles.size()];
	float * light_power_cumulative = new float[light_triangles.size()];

	for (int m = 0; m < light_meshes.size(); m++) {
		LightMesh & light_mesh = light_meshes[m];

		float cumulative_power = 0.0f;

		for (int i = light_mesh.triangle_first_index; i < light_mesh.triangle_first_index + light_mesh.triangle_count; i++) {
			light_indices[i] = light_triangles[i].index;

			cumulative_power += light_triangles[i].power;
			light_power_cumulative[i] = cumulative_power;
		}

		light_mesh.power = cumulative_power;
	}

	if (light_triangles.size() == 0) return;

	ptr_light_indices          = CUDAMemory::malloc(light_indices,          light_triangles.size());
	ptr_light_areas_cumulative = CUDAMemory::malloc(light_power_cumulative, light_triangles.size());

	cuda_module.get_global("light_indices")         .set_value(ptr_light_indices);
	cuda_module.get_global("light_power_cumulative").set_value(ptr_light_areas_cumulative);

	delete [] light_indices;
	delete [] light_power_cumulative;

	float * light_mesh_power_unscaled       = MALLOCA(float, scene.meshes.size());
	int   * light_mesh_triangle_count       = MALLOCA(int,   scene.meshes.size());
	int   * light_mesh_triangle_first_index = MALLOCA(int,   scene.meshes.size());
	
	int light_mesh_count  = 0;
		
	for (int m = 0; m < scene.meshes.size(); m++) {
		int light_mesh_data_index = light_mesh_data_indices[scene.meshes[m].mesh_data_handle.handle];

		if (light_mesh_data_index != INVALID) {
			const LightMesh & light_mesh = light_meshes[light_mesh_data_index];
			
			int mesh_index = light_mesh_count++;
			assert(mesh_index < scene.meshes.size());

			scene.meshes[m].light_index = mesh_index;
			scene.meshes[m].light_power = light_mesh.power;

			light_mesh_power_unscaled      [mesh_index] = light_mesh.power;
			light_mesh_triangle_first_index[mesh_index] = light_mesh.triangle_first_index;
			light_mesh_triangle_count      [mesh_index] = light_mesh.triangle_count;
		} else {
			scene.meshes[m].light_index = INVALID;
		}
	}

	if (ptr_light_mesh_power_unscaled      .ptr != NULL) CUDAMemory::free(ptr_light_mesh_power_unscaled);
	if (ptr_light_mesh_triangle_count      .ptr != NULL) CUDAMemory::free(ptr_light_mesh_triangle_count);
	if (ptr_light_mesh_triangle_first_index.ptr != NULL) CUDAMemory::free(ptr_light_mesh_triangle_first_index);
	if (ptr_light_mesh_power_scaled        .ptr != NULL) CUDAMemory::free(ptr_light_mesh_power_scaled);
	if (ptr_light_mesh_transform_indices   .ptr != NULL) CUDAMemory::free(ptr_light_mesh_transform_indices);

	ptr_light_mesh_power_unscaled       = CUDAMemory::malloc(light_mesh_power_unscaled,       light_mesh_count);    
	ptr_light_mesh_triangle_count       = CUDAMemory::malloc(light_mesh_triangle_count,       light_mesh_count);
	ptr_light_mesh_triangle_first_index = CUDAMemory::malloc(light_mesh_triangle_first_index, light_mesh_count);

	cuda_module.get_global("light_mesh_power_unscaled")      .set_value(ptr_light_mesh_power_unscaled);
	cuda_module.get_global("light_mesh_triangle_count")      .set_value(ptr_light_mesh_triangle_count);
	cuda_module.get_global("light_mesh_triangle_first_index").set_value(ptr_light_mesh_triangle_first_index);

	// These pointers are only filled in and copied to the GPU once the TLAS is constructed, 
	// therefore the scene_invalidated flag is required to be set.
	ptr_light_mesh_power_scaled      = CUDAMemory::malloc<float>(light_mesh_count);
	ptr_light_mesh_transform_indices = CUDAMemory::malloc<int>  (light_mesh_count);
	invalidated_scene = true;

	cuda_module.get_global("light_mesh_power_scaled")     .set_value(ptr_light_mesh_power_scaled);
	cuda_module.get_global("light_mesh_transform_indices").set_value(ptr_light_mesh_transform_indices);

	FREEA(light_mesh_power_unscaled);
	FREEA(light_mesh_triangle_count);
	FREEA(light_mesh_triangle_first_index);

	FREEA(light_mesh_data_indices);
}

// Construct Top Level Acceleration Structure (TLAS) over the Meshes in the Scene
void Pathtracer::build_tlas() {
	tlas_bvh_builder.build(scene.meshes.data(), scene.meshes.size());

	tlas.index_count = tlas_raw.index_count;
	tlas.indices     = tlas_raw.indices;
	tlas.node_count  = tlas_raw.node_count;

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	const BVH & tlas = tlas_raw;
#else
	tlas_converter.build(tlas_raw);
#endif

	assert(tlas.index_count == scene.meshes.size());
	CUDAMemory::memcpy(ptr_bvh_nodes, tlas.nodes, tlas.node_count);

	int   light_mesh_count   = 0;
	float lights_total_power = 0.0f;

	for (int i = 0; i < scene.meshes.size(); i++) {
		const Mesh & mesh = scene.meshes[tlas.indices[i]];

		pinned_mesh_bvh_root_indices[i] = mesh_data_bvh_offsets[mesh.mesh_data_handle.handle] | (mesh.has_identity_transform() << 31);

		assert(mesh.material_handle.handle != INVALID);
		pinned_mesh_material_ids[i] = mesh.material_handle.handle;

		memcpy(pinned_mesh_transforms     [i].cells, mesh.transform     .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_inv [i].cells, mesh.transform_inv .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_prev[i].cells, mesh.transform_prev.cells, sizeof(Matrix3x4));

		bool mesh_is_light = mesh.light_index != INVALID;
		if (mesh_is_light) {
			float light_power_scaled = mesh.light_power * mesh.scale * mesh.scale;

			assert(mesh.light_index < scene.meshes.size());

			pinned_light_mesh_transform_indices[mesh.light_index] = i;
			pinned_light_mesh_area_scaled      [mesh.light_index] = light_power_scaled;
			
			light_mesh_count++;
			lights_total_power += light_power_scaled;
		}
	}

	CUDAMemory::memcpy(ptr_mesh_bvh_root_indices,  pinned_mesh_bvh_root_indices, scene.meshes.size());
	CUDAMemory::memcpy(ptr_mesh_material_ids,      pinned_mesh_material_ids,     scene.meshes.size());
	CUDAMemory::memcpy(ptr_mesh_transforms,        pinned_mesh_transforms,       scene.meshes.size());
	CUDAMemory::memcpy(ptr_mesh_transforms_inv,    pinned_mesh_transforms_inv,   scene.meshes.size());
	CUDAMemory::memcpy(ptr_mesh_transforms_prev,   pinned_mesh_transforms_prev,  scene.meshes.size());
	
	if (scene.has_lights) {
		CUDAMemory::memcpy(ptr_light_mesh_transform_indices, pinned_light_mesh_transform_indices, light_mesh_count);
		CUDAMemory::memcpy(ptr_light_mesh_power_scaled,      pinned_light_mesh_area_scaled,       light_mesh_count);
	}

	global_lights_total_power.set_value(0.999f * lights_total_power);
}

void Pathtracer::update(float delta) {
	if (invalidated_settings && settings.enable_svgf && scene.camera.aperture_radius > 0.0f) {
		puts("WARNING: SVGF and DoF cannot simultaneously be enabled!");
		scene.camera.aperture_radius = 0.0f;
	}

	if (invalidated_materials) {
		const std::vector<Material> & materials = scene.asset_manager.materials;

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
						log2f(material.transmittance.x), 
						log2f(material.transmittance.y), 
						log2f(material.transmittance.z)
					);
					cuda_materials[i].dielectric.index_of_refraction = Math::max(material.index_of_refraction, 1.0001f);
					break;
				}
				case Material::Type::GLOSSY: {
					cuda_materials[i].glossy.diffuse             = material.diffuse;
					cuda_materials[i].glossy.texture_id          = material.texture_id.handle;
					cuda_materials[i].glossy.index_of_refraction = Math::max(material.index_of_refraction, 1.0001f);
					cuda_materials[i].glossy.roughness           = Math::max(material.linear_roughness * material.linear_roughness, 1e-6f);
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
				ray_buffer_shade_diffuse.init(batch_size);
			} else {
				ray_buffer_shade_diffuse.free();
			}
			global_ray_buffer_shade_diffuse.set_value(ray_buffer_shade_diffuse);
		}
		
		// Handle (dis)appearance of Dielectric OR Glossy materials (they share the same Material buffer)
		if (dielectric_or_glossy_changed) {
			if (scene.has_dielectric || scene.has_glossy) {
				ray_buffer_shade_dielectric_and_glossy.init(batch_size);
			} else {
				ray_buffer_shade_dielectric_and_glossy.free();
			}
			global_ray_buffer_shade_dielectric_and_glossy.set_value(ray_buffer_shade_dielectric_and_glossy);
		}

		// Handle (dis)appearance of Light materials
		if (lights_changed) {	
			if (scene.has_lights) {
				ray_buffer_shadow.init(batch_size);

				invalidated_scene = true;
			} else {
				ray_buffer_shadow.free();

				CUDAMemory::free(ptr_light_mesh_power_unscaled);
				CUDAMemory::free(ptr_light_mesh_triangle_count);
				CUDAMemory::free(ptr_light_mesh_triangle_first_index);
				
				CUDAMemory::free(ptr_light_mesh_power_scaled);
				CUDAMemory::free(ptr_light_mesh_transform_indices);
				
				global_lights_total_power.set_value(0.0f);
				
				for (int i = 0; i < scene.meshes.size(); i++) {
					scene.meshes[i].light_index = INVALID;
				}
			}

			global_ray_buffer_shadow.set_value(ray_buffer_shadow);
		}

		if (scene.has_lights) calc_light_power();

		frames_accumulated = 0;
		invalidated_materials = false;
	}

	if (settings.enable_scene_update) {
		scene.update(delta);
		invalidated_scene = true;
	} else if (settings.enable_svgf || invalidated_scene) {
		scene.camera.update(0.0f, settings);
		scene.update(0.0f);
	} 

	if (invalidated_scene) {
		build_tlas();

		// If SVGF is enabled we can handle Scene updates using reprojection,
		// otherwise 'frames_accumulated' needs to be reset in order to avoid ghosting
		if (!settings.enable_svgf) {
			frames_accumulated = 0;
		}

		invalidated_scene = false;
	}
	
	scene.camera.update(delta, settings);
	
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
		
		if (!settings.enable_svgf) {
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

	if (settings.enable_svgf) {
		struct SVGFData {
			alignas(16) Matrix4 view_projection;
			alignas(16) Matrix4 view_projection_prev;
		} svgf_data;

		svgf_data.view_projection      = scene.camera.view_projection;
		svgf_data.view_projection_prev = scene.camera.view_projection_prev;

		global_svgf_data.set_value(svgf_data);
	}

	if (invalidated_settings) {
		frames_accumulated = 0;

		global_settings.set_value(settings);
	} else if (settings.enable_svgf) {
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

	// Render in batches of BATCH_SIZE pixels at a time
	while (pixels_left > 0) {
		int pixel_offset = pixel_count - pixels_left;
		int pixel_count  = Math::min(batch_size, pixels_left);

		event_pool.record(event_desc_primary);

		// Generate primary Rays from the current Camera orientation
		kernel_generate.execute(
			Random::get_value(),
			frames_accumulated,
			pixel_offset,
			pixel_count
		);

		for (int bounce = 0; bounce < settings.num_bounces; bounce++) {
			// Extend all Rays that are still alive to their next Triangle intersection
			event_pool.record(event_desc_trace[bounce]);
			kernel_trace.execute(bounce);
			
			event_pool.record(event_desc_sort[bounce]);
			kernel_sort.execute(Random::get_value(), bounce);

			// Process the various Material types in different Kernels
			if (scene.has_diffuse) {
				event_pool.record(event_desc_shade_diffuse[bounce]);
				kernel_shade_diffuse.execute(Random::get_value(), bounce, frames_accumulated);
			}

			if (scene.has_dielectric) {
				event_pool.record(event_desc_shade_dielectric[bounce]);
				kernel_shade_dielectric.execute(Random::get_value(), bounce);
			}

			if (scene.has_glossy) {
				event_pool.record(event_desc_shade_glossy[bounce]);
				kernel_shade_glossy.execute(Random::get_value(), bounce, frames_accumulated);
			}

			// Trace shadow Rays
			if (scene.has_lights && settings.enable_next_event_estimation) {
				event_pool.record(event_desc_shadow_trace[bounce]);
				kernel_trace_shadow.execute(bounce);
			}
		}

		pixels_left -= batch_size;

		if (pixels_left > 0) {
			// Set buffer sizes to appropriate pixel count for next Batch
			pinned_buffer_sizes->reset(Math::min(batch_size, pixels_left));
			global_buffer_sizes.set_value(*pinned_buffer_sizes);
		}
	}

	if (settings.enable_svgf) {
		// Temporal reprojection + integration
		event_pool.record(event_desc_svgf_reproject);
		kernel_svgf_reproject.execute(frames_accumulated);

		CUdeviceptr direct_in    = ptr_frame_buffer_direct    .ptr;
		CUdeviceptr direct_out   = ptr_frame_buffer_direct_alt.ptr;
		CUdeviceptr indirect_in  = ptr_frame_buffer_indirect    .ptr;
		CUdeviceptr indirect_out = ptr_frame_buffer_indirect_alt.ptr;

		if (settings.enable_spatial_variance) {
			// Estimate Variance spatially
			event_pool.record(event_desc_svgf_variance);
			kernel_svgf_variance.execute(direct_in, indirect_in, direct_out, indirect_out);

			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);
		}

		// À-Trous Filter
		for (int i = 0; i < settings.atrous_iterations; i++) {
			int step_size = 1 << i;
			
			event_pool.record(event_desc_svgf_atrous[i]);
			kernel_svgf_atrous.execute(direct_in, indirect_in, direct_out, indirect_out, step_size);

			// Ping-Pong the Frame Buffers
			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);
		}

		event_pool.record(event_desc_svgf_finalize);
		kernel_svgf_finalize.execute(direct_in, indirect_in);

		if (settings.enable_taa) {
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
	
	if (settings.modulate_albedo) CUDAMemory::memset_async(ptr_frame_buffer_albedo, 0, screen_pitch * screen_height, stream_memset);
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
