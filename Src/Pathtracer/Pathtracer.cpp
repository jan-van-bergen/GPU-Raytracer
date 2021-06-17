#include "Pathtracer.h"

#include <algorithm>

#include <GL/glew.h>

#include "CUDA/CUDAContext.h"

#include "Assets/MeshData.h"
#include "Assets/Material.h"

#include "Math/Vector4.h"

#include "Util/Random.h"
#include "Util/BlueNoise.h"

#include "Util/Util.h"
#include "Util/ScopeTimer.h"

void Pathtracer::init(int mesh_count, char const ** mesh_names, char const * sky_name, unsigned frame_buffer_handle) {
	scene.init(mesh_count, mesh_names, sky_name);

	cuda_init(frame_buffer_handle, SCREEN_WIDTH, SCREEN_HEIGHT);
}

void Pathtracer::cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) {
	// Init CUDA Module and its Kernels
	module.init("CUDA_Source/Pathtracer.cu", CUDAContext::compute_capability, MAX_REGISTERS);
	
	kernel_generate        .init(&module, "kernel_generate");
	kernel_trace           .init(&module, "kernel_trace");
	kernel_sort            .init(&module, "kernel_sort");
	kernel_shade_diffuse   .init(&module, "kernel_shade_diffuse");
	kernel_shade_dielectric.init(&module, "kernel_shade_dielectric");
	kernel_shade_glossy    .init(&module, "kernel_shade_glossy");
	kernel_trace_shadow    .init(&module, "kernel_trace_shadow");
	kernel_svgf_reproject  .init(&module, "kernel_svgf_reproject");
	kernel_svgf_variance   .init(&module, "kernel_svgf_variance");
	kernel_svgf_atrous     .init(&module, "kernel_svgf_atrous");
	kernel_svgf_finalize   .init(&module, "kernel_svgf_finalize");
	kernel_taa             .init(&module, "kernel_taa");
	kernel_taa_finalize    .init(&module, "kernel_taa_finalize");
	kernel_accumulate      .init(&module, "kernel_accumulate");

	// Set Block dimensions for all Kernels
	kernel_svgf_reproject.occupancy_max_block_size_2d();
	kernel_svgf_variance .occupancy_max_block_size_2d();
	kernel_svgf_atrous   .occupancy_max_block_size_2d();
	kernel_svgf_finalize .occupancy_max_block_size_2d();
	kernel_taa           .occupancy_max_block_size_2d();
	kernel_taa_finalize  .occupancy_max_block_size_2d();
	kernel_accumulate    .occupancy_max_block_size_2d();

	kernel_generate        .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_sort            .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_diffuse   .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_dielectric.set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_glossy    .set_block_dim(WARP_SIZE * 2, 1, 1);
	
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

	resize_init(frame_buffer_handle, screen_width, screen_height);
	
	// Set global Material table
	ptr_materials = CUDAMemory::malloc(scene.materials);
	module.get_global("materials").set_value(ptr_materials);
	
	scene.wait_until_textures_loaded();

	// Set global Texture table
	int texture_count = scene.textures.size();
	if (texture_count > 0) {
		tex_objects = new CUtexObject     [texture_count];
		tex_arrays  = new CUmipmappedArray[texture_count];
		
		// Get maximum anisotropy from OpenGL
		int max_aniso; glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_aniso);

		for (int i = 0; i < texture_count; i++) {
			const Texture & texture = scene.textures[i];

			// Create mipmapped CUDA array
			tex_arrays[i] = CUDAMemory::create_array_mipmap(
				texture.width,
				texture.height,
				texture.channels,
				texture.get_cuda_array_format(),
				texture.mip_levels
			);

			// Upload each level of the mipmap
			for (int level = 0; level < texture.mip_levels; level++) {
				CUarray level_array;
				CUDACALL(cuMipmappedArrayGetLevel(&level_array, tex_arrays[i], level));

				int level_width_in_bytes = texture.get_width_in_bytes(level);
				int level_height         = texture.height >> level;

				CUDAMemory::copy_array(level_array, level_width_in_bytes, level_height, texture.data + texture.mip_offsets[level]);
			}

			// Describe the Array to read from
			CUDA_RESOURCE_DESC res_desc = { };
			res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
			res_desc.res.mipmap.hMipmappedArray = tex_arrays[i];

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

			CUDACALL(cuTexObjectCreate(tex_objects + i, &res_desc, &tex_desc, &view_desc));
		}

		ptr_textures = CUDAMemory::malloc(tex_objects, texture_count);
		module.get_global("textures").set_value(ptr_textures);
	}

	int mesh_data_count = scene.mesh_datas.size();

	mesh_data_bvh_offsets = new int[mesh_data_count];

	int * mesh_data_index_offsets    = MALLOCA(int, mesh_data_count);
	int * mesh_data_triangle_offsets = MALLOCA(int, mesh_data_count);

	int global_bvh_node_count = 2 * scene.mesh_count; // Reserve 2 times Mesh count for TLAS
	int global_index_count    = 0;
	int global_triangle_count = 0;

	for (int i = 0; i < mesh_data_count; i++) {
		mesh_data_bvh_offsets     [i] = global_bvh_node_count;
		mesh_data_index_offsets   [i] = global_index_count;
		mesh_data_triangle_offsets[i] = global_triangle_count;

		global_bvh_node_count += scene.mesh_datas[i]->bvh.node_count;
		global_index_count    += scene.mesh_datas[i]->bvh.index_count;
		global_triangle_count += scene.mesh_datas[i]->triangle_count;
	}

	BVHNodeType * global_bvh_nodes = new BVHNodeType[global_bvh_node_count];
	int         * global_indices   = new int        [global_index_count];
	Triangle    * global_triangles = new Triangle   [global_triangle_count];

	for (int m = 0; m < mesh_data_count; m++) {
		const MeshData * mesh_data = scene.mesh_datas[m];

		for (int n = 0; n < mesh_data->bvh.node_count; n++) {
			BVHNodeType & node = global_bvh_nodes[mesh_data_bvh_offsets[m] + n];

			node = mesh_data->bvh.nodes[n];

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

		for (int i = 0; i < mesh_data->bvh.index_count; i++) {
			global_indices[mesh_data_index_offsets[m] + i] = mesh_data->bvh.indices[i] + mesh_data_triangle_offsets[m];
		}

		for (int t = 0; t < mesh_data->triangle_count; t++) {
			global_triangles[mesh_data_triangle_offsets[m] + t]              = mesh_data->triangles[t];
			global_triangles[mesh_data_triangle_offsets[m] + t].material_id += mesh_data->material_offset;
		}
	}

	pinned_mesh_bvh_root_indices        = CUDAMemory::malloc_pinned<int>      (scene.mesh_count);
	pinned_mesh_transforms              = CUDAMemory::malloc_pinned<Matrix3x4>(scene.mesh_count);
	pinned_mesh_transforms_inv          = CUDAMemory::malloc_pinned<Matrix3x4>(scene.mesh_count);
	pinned_mesh_transforms_prev         = CUDAMemory::malloc_pinned<Matrix3x4>(scene.mesh_count);
	pinned_light_mesh_transform_indices = CUDAMemory::malloc_pinned<int>      (scene.mesh_count);
	pinned_light_mesh_area_scaled       = CUDAMemory::malloc_pinned<float>    (scene.mesh_count);

	ptr_mesh_bvh_root_indices = CUDAMemory::malloc<int>      (scene.mesh_count);
	ptr_mesh_transforms       = CUDAMemory::malloc<Matrix3x4>(scene.mesh_count);
	ptr_mesh_transforms_inv   = CUDAMemory::malloc<Matrix3x4>(scene.mesh_count);
	ptr_mesh_transforms_prev  = CUDAMemory::malloc<Matrix3x4>(scene.mesh_count);

	module.get_global("mesh_bvh_root_indices").set_value(ptr_mesh_bvh_root_indices);
	module.get_global("mesh_transforms")      .set_value(ptr_mesh_transforms);
	module.get_global("mesh_transforms_inv")  .set_value(ptr_mesh_transforms_inv);
	module.get_global("mesh_transforms_prev") .set_value(ptr_mesh_transforms_prev);
	
	ptr_bvh_nodes = CUDAMemory::malloc<BVHNodeType>(global_bvh_nodes, global_bvh_node_count);

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	module.get_global("bvh_nodes").set_value(ptr_bvh_nodes);
#elif BVH_TYPE == BVH_QBVH
	module.get_global("qbvh_nodes").set_value(ptr_bvh_nodes);
#elif BVH_TYPE == BVH_CWBVH
	module.get_global("cwbvh_nodes").set_value(ptr_bvh_nodes);
#endif

	tlas_bvh_builder.init(&tlas_raw, scene.mesh_count, 1);

	tlas_raw.node_count = scene.mesh_count * 2;
#if BVH_TYPE == BVH_QBVH || BVH_TYPE == BVH_CWBVH
	tlas_converter.init(&tlas, tlas_raw);
#endif

	CUDATriangle * triangles             = new CUDATriangle[global_index_count];
	int          * triangle_material_ids = new int         [global_index_count];
	float        * triangle_lods         = new float       [global_index_count];

	int * reverse_indices = new int[global_index_count];

	for (int i = 0; i < global_index_count; i++) {
		int index = global_indices[i];

		assert(index < global_triangle_count);

		triangles[i].position_0      = global_triangles[index].position_0;
		triangles[i].position_edge_1 = global_triangles[index].position_1 - global_triangles[index].position_0;
		triangles[i].position_edge_2 = global_triangles[index].position_2 - global_triangles[index].position_0;

		triangles[i].normal_0      = global_triangles[index].normal_0;
		triangles[i].normal_edge_1 = global_triangles[index].normal_1 - global_triangles[index].normal_0;
		triangles[i].normal_edge_2 = global_triangles[index].normal_2 - global_triangles[index].normal_0;

		triangles[i].tex_coord_0      = global_triangles[index].tex_coord_0;
		triangles[i].tex_coord_edge_1 = global_triangles[index].tex_coord_1 - global_triangles[index].tex_coord_0;
		triangles[i].tex_coord_edge_2 = global_triangles[index].tex_coord_2 - global_triangles[index].tex_coord_0;

		int material_id = global_triangles[index].material_id;
		triangle_material_ids[i] = material_id;

		int texture_id = scene.materials[material_id].texture_id;
		if (texture_id != INVALID) {
			const Texture & texture = scene.textures[texture_id];

			// Triangle texture base LOD as described in "Texture Level of Detail Strategies for Real-Time Ray Tracing"
			float t_a = float(texture.width * texture.height) * fabsf(
				triangles[i].tex_coord_edge_1.x * triangles[i].tex_coord_edge_2.y -
				triangles[i].tex_coord_edge_2.x * triangles[i].tex_coord_edge_1.y
			); 
			float p_a = Vector3::length(Vector3::cross(triangles[i].position_edge_1, triangles[i].position_edge_2));

			triangle_lods[i] = 0.5f * log2f(t_a / p_a);
		} else {
			triangle_lods[i] = 0.0f;
		}

		reverse_indices[index] = i;
	}

	ptr_triangles             = CUDAMemory::malloc(triangles,             global_index_count);
	ptr_triangle_material_ids = CUDAMemory::malloc(triangle_material_ids, global_index_count);
	ptr_triangle_lods         = CUDAMemory::malloc(triangle_lods,         global_index_count);

	module.get_global("triangles")            .set_value(ptr_triangles);
	module.get_global("triangle_material_ids").set_value(ptr_triangle_material_ids);
	module.get_global("triangle_lods")        .set_value(ptr_triangle_lods);
	
	if (scene.has_lights) {
		// Initialize Lights
		struct LightTriangle {
			int   index;
			float area;
		};
		std::vector<LightTriangle> light_triangles;

		struct LightMesh {
			int triangle_first_index;
			int triangle_count;

			float area;
		};
		std::vector<LightMesh> light_meshes;

		int * light_mesh_data_indices = MALLOCA(int, mesh_data_count);
		memset(light_mesh_data_indices, -1, mesh_data_count * sizeof(int));

		// Loop over every MeshData and check whether it has at least 1 Triangle that is a Light
		for (int m = 0; m < mesh_data_count; m++) {
			const MeshData * mesh_data = scene.mesh_datas[m];

			LightMesh * light_mesh = nullptr;

			// For every Triangle, check whether it is a Light based on its Material
			for (int t = 0; t < mesh_data->triangle_count; t++) {
				const Triangle & triangle = mesh_data->triangles[t];

				if (scene.materials[mesh_data->material_offset + triangle.material_id].type == Material::Type::LIGHT) {
					float area = 0.5f * Vector3::length(Vector3::cross(
						triangle.position_1 - triangle.position_0,
						triangle.position_2 - triangle.position_0
					));

					if (light_mesh == nullptr) {
						light_mesh_data_indices[m] = light_meshes.size();

						light_mesh = &light_meshes.emplace_back();
						light_mesh->triangle_first_index = light_triangles.size();
						light_mesh->triangle_count = 0;
					}

					light_triangles.push_back({ reverse_indices[mesh_data_triangle_offsets[m] + t], area });

					light_mesh->triangle_count++;
				}
			}

			if (light_mesh) {		
				// Sort Lights on area within each Mesh
				LightTriangle * triangles_begin = light_triangles.data() + light_mesh->triangle_first_index;
				LightTriangle * triangles_end   = triangles_begin        + light_mesh->triangle_count;

				assert(triangles_end > triangles_begin);

				std::sort(triangles_begin, triangles_end, [](const LightTriangle & a, const LightTriangle & b) { return a.area < b.area; });
			}
		}

		int   * light_indices          = new int  [light_triangles.size()];
		float * light_areas_cumulative = new float[light_triangles.size()];

		for (int m = 0; m < light_meshes.size(); m++) {
			LightMesh & light_mesh = light_meshes[m];

			float cumulative_area = 0.0f;

			for (int i = light_mesh.triangle_first_index; i < light_mesh.triangle_first_index + light_mesh.triangle_count; i++) {
				light_indices[i] = light_triangles[i].index;

				cumulative_area += light_triangles[i].area;
				light_areas_cumulative[i] = cumulative_area;
			}

			light_mesh.area = cumulative_area;
		}

		ptr_light_indices          = CUDAMemory::malloc(light_indices,          light_triangles.size());
		ptr_light_areas_cumulative = CUDAMemory::malloc(light_areas_cumulative, light_triangles.size());

		module.get_global("light_indices")         .set_value(ptr_light_indices);
		module.get_global("light_areas_cumulative").set_value(ptr_light_areas_cumulative);

		delete [] light_indices;
		delete [] light_areas_cumulative;

		float * light_mesh_area_unscaled        = MALLOCA(float, scene.mesh_count);
		int   * light_mesh_triangle_count       = MALLOCA(int,   scene.mesh_count);
		int   * light_mesh_triangle_first_index = MALLOCA(int,   scene.mesh_count);
		
		int light_total_count = 0;
		int light_mesh_count  = 0;
		
		for (int m = 0; m < scene.mesh_count; m++) {
			int light_mesh_data_index = light_mesh_data_indices[scene.meshes[m].mesh_data_index];

			if (light_mesh_data_index != -1) {
				const LightMesh & light_mesh = light_meshes[light_mesh_data_index];

				scene.meshes[m].light_index = light_mesh_count;
				scene.meshes[m].light_area  = light_mesh.area;

				int mesh_index = light_mesh_count++;
				assert(mesh_index < scene.mesh_count);

				light_mesh_area_unscaled       [mesh_index] = light_mesh.area;
				light_mesh_triangle_first_index[mesh_index] = light_mesh.triangle_first_index;
				light_mesh_triangle_count      [mesh_index] = light_mesh.triangle_count;

				light_total_count += light_mesh.triangle_count;
			}
		}
		
		module.get_global("light_total_count_inv").set_value(1.0f / float(light_total_count));
		module.get_global("light_mesh_count")     .set_value(light_mesh_count);

		ptr_light_mesh_area_unscaled        = CUDAMemory::malloc(light_mesh_area_unscaled,        light_mesh_count);    
		ptr_light_mesh_triangle_count       = CUDAMemory::malloc(light_mesh_triangle_count,       light_mesh_count);
		ptr_light_mesh_triangle_first_index = CUDAMemory::malloc(light_mesh_triangle_first_index, light_mesh_count);

		module.get_global("light_mesh_area_unscaled")       .set_value(ptr_light_mesh_area_unscaled);
		module.get_global("light_mesh_triangle_count")      .set_value(ptr_light_mesh_triangle_count);
		module.get_global("light_mesh_triangle_first_index").set_value(ptr_light_mesh_triangle_first_index);

		ptr_light_total_area = module.get_global("light_total_area").ptr;
		ptr_light_mesh_area_scaled       = CUDAMemory::malloc<float>(light_mesh_count);
		ptr_light_mesh_transform_indices = CUDAMemory::malloc<int>  (light_mesh_count);

		module.get_global("light_mesh_area_scaled")      .set_value(ptr_light_mesh_area_scaled);
		module.get_global("light_mesh_transform_indices").set_value(ptr_light_mesh_transform_indices);

		FREEA(light_mesh_area_unscaled);
		FREEA(light_mesh_triangle_count);
		FREEA(light_mesh_triangle_first_index);

		FREEA(light_mesh_data_indices);
	} else {
		module.get_global("light_total_count_inv").set_value(INFINITY); // 1 / 0
	}

	delete [] global_bvh_nodes;
	delete [] global_indices;
	delete [] global_triangles;

	delete [] triangles;
	delete [] triangle_lods;
	delete [] triangle_material_ids;

	delete [] reverse_indices;

	ptr_sky_data = CUDAMemory::malloc(scene.sky.data, scene.sky.width * scene.sky.height);

	module.get_global("sky_width") .set_value(scene.sky.width);
	module.get_global("sky_height").set_value(scene.sky.height);
	module.get_global("sky_data")  .set_value(ptr_sky_data);
	
	// Set Blue Noise Sampler globals
	ptr_sobol_256spp_256d = CUDAMemory::malloc(sobol_256spp_256d);
	ptr_scrambling_tile   = CUDAMemory::malloc(scrambling_tile);
	ptr_ranking_tile      = CUDAMemory::malloc(ranking_tile);

	module.get_global("sobol_256spp_256d").set_value(ptr_sobol_256spp_256d);
	module.get_global("scrambling_tile")  .set_value(ptr_scrambling_tile);
	module.get_global("ranking_tile")     .set_value(ptr_ranking_tile);
	
	// Initialize buffers used by Wavefront kernels
	                                                  ray_buffer_trace                      .init(batch_size);
	/*if (scene.has_diffuse)                       */ ray_buffer_shade_diffuse              .init(batch_size);
	/*if (scene.has_dielectric || scene.has_glossy)*/ ray_buffer_shade_dielectric_and_glossy.init(batch_size);
	/*if (scene.has_lights)                        */ ray_buffer_shadow                     .init(batch_size);

	module.get_global("ray_buffer_trace")                      .set_value(ray_buffer_trace);
	module.get_global("ray_buffer_shade_diffuse")              .set_value(ray_buffer_shade_diffuse);
	module.get_global("ray_buffer_shade_dielectric_and_glossy").set_value(ray_buffer_shade_dielectric_and_glossy);
	module.get_global("ray_buffer_shadow")                     .set_value(ray_buffer_shadow);

	pinned_buffer_sizes = CUDAMemory::malloc_pinned<BufferSizes>();
	pinned_buffer_sizes->reset(batch_size);
	
	global_buffer_sizes = module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	global_camera    = module.get_global("camera");
	global_settings  = module.get_global("settings");
	global_svgf_data = module.get_global("svgf_data");

	// Initialize timers
	int display_order = 0;
	event_info_primary = { display_order++, "Primary", "Primary" };

	for (int i = 0; i < MAX_BOUNCES; i++) {
		const int len = 16;
		char    * category = new char[len];
		sprintf_s(category, len, "Bounce %i", i);

		event_info_trace           [i] = { display_order, category, "Trace" };
		event_info_sort            [i] = { display_order, category, "Sort" };
		event_info_shade_diffuse   [i] = { display_order, category, "Diffuse" };
		event_info_shade_dielectric[i] = { display_order, category, "Dielectric" };
		event_info_shade_glossy    [i] = { display_order, category, "Glossy" };
		event_info_shadow_trace    [i] = { display_order, category, "Shadow" };

		display_order++;
	}

	event_info_svgf_reproject = { display_order, "SVGF", "Reproject" };
	event_info_svgf_variance  = { display_order, "SVGF", "Variance" };

	for (int i = 0; i < MAX_ATROUS_ITERATIONS; i++) {
		const int len = 16;
		char    * name = new char[len];
		sprintf_s(name, len, "A Trous %i", i);

		event_info_svgf_atrous[i] = { display_order, "SVGF", name };
	}
	event_info_svgf_finalize = { display_order++, "SVGF", "Finalize" };

	event_info_taa         = { display_order, "Post", "TAA" };
	event_info_reconstruct = { display_order, "Post", "Reconstruct" };
	event_info_accumulate  = { display_order, "Post", "Accumulate" };

	event_info_end = { ++display_order, "END", "END" };
	
	// Realloc as pinned memory
	delete [] tlas.nodes;
	tlas.nodes = CUDAMemory::malloc_pinned<BVHNodeType>(2 * scene.mesh_count);
	
	scene.camera.update(0.0f, settings);
	scene.update(0.0f);

	scene_invalidated = true;

	unsigned long long bytes_available = CUDAContext::get_available_memory();
	unsigned long long bytes_allocated = CUDAContext::total_memory - bytes_available;

	printf("CUDA Memory allocated: %8llu KB (%6llu MB)\n",   bytes_allocated >> 10, bytes_allocated >> 20);
	printf("CUDA Memory free:      %8llu KB (%6llu MB)\n\n", bytes_available >> 10, bytes_available >> 20);
}

void Pathtracer::cuda_free() {
	CUDAMemory::free(ptr_materials);

	if (scene.textures.size() > 0) {
		CUDAMemory::free(ptr_textures);

		for (int i = 0; i < scene.textures.size(); i++) {
			CUDAMemory::free_array(tex_arrays[i]);
			CUDAMemory::free_texture(tex_objects[i]);
		}
	
		delete [] tex_objects;
		delete [] tex_arrays;
	}

	CUDAMemory::free_pinned(pinned_buffer_sizes);
	CUDAMemory::free_pinned(pinned_mesh_bvh_root_indices);
	CUDAMemory::free_pinned(pinned_mesh_transforms);
	CUDAMemory::free_pinned(pinned_mesh_transforms_inv);
	CUDAMemory::free_pinned(pinned_mesh_transforms_prev);
	CUDAMemory::free_pinned(pinned_light_mesh_transform_indices);
	CUDAMemory::free_pinned(pinned_light_mesh_area_scaled);

	CUDAMemory::free(ptr_mesh_bvh_root_indices);
	CUDAMemory::free(ptr_mesh_transforms);
	CUDAMemory::free(ptr_mesh_transforms_inv);
	CUDAMemory::free(ptr_mesh_transforms_prev);

	CUDAMemory::free(ptr_bvh_nodes);

	CUDAMemory::free(ptr_triangles);
	CUDAMemory::free(ptr_triangle_material_ids);
	CUDAMemory::free(ptr_triangle_lods);

	CUDAMemory::free(ptr_light_indices);
	CUDAMemory::free(ptr_light_areas_cumulative);
	CUDAMemory::free(ptr_light_mesh_area_unscaled);
	CUDAMemory::free(ptr_light_mesh_triangle_count);
	CUDAMemory::free(ptr_light_mesh_triangle_first_index);
	CUDAMemory::free(ptr_light_mesh_area_scaled);
	CUDAMemory::free(ptr_light_mesh_transform_indices);

	CUDAMemory::free(ptr_sky_data);

	CUDAMemory::free(ptr_sobol_256spp_256d);
	CUDAMemory::free(ptr_scrambling_tile);
	CUDAMemory::free(ptr_ranking_tile);
	                                              ray_buffer_trace                      .free();
	if (scene.has_diffuse)                        ray_buffer_shade_diffuse              .free();
	if (scene.has_dielectric || scene.has_glossy) ray_buffer_shade_dielectric_and_glossy.free();
	if (scene.has_lights)                         ray_buffer_shadow                     .free();

	tlas_bvh_builder.free();
#if BVH_TYPE == BVH_QBVH || BVH_TYPE == BVH_CWBVH
	tlas_converter.free();
#endif

	resize_free();

	module.free();
}

void Pathtracer::resize_init(unsigned frame_buffer_handle, int width, int height) {
	screen_width  = width;
	screen_height = height;
	screen_pitch  = Math::divide_round_up(width, WARP_SIZE) * WARP_SIZE;

	pixel_count = width * height;
	batch_size  = Math::min(BATCH_SIZE, pixel_count);

	module.get_global("screen_width") .set_value(screen_width);
	module.get_global("screen_pitch") .set_value(screen_pitch);
	module.get_global("screen_height").set_value(screen_height);

	// Allocate GBuffers
	array_gbuffer_normal_and_depth        = CUDAMemory::create_array(screen_pitch, height, 4, CU_AD_FORMAT_FLOAT);
	array_gbuffer_mesh_id_and_triangle_id = CUDAMemory::create_array(screen_pitch, height, 2, CU_AD_FORMAT_SIGNED_INT32);
	array_gbuffer_screen_position_prev    = CUDAMemory::create_array(screen_pitch, height, 2, CU_AD_FORMAT_FLOAT);

	surf_gbuffer_normal_and_depth        = CUDAMemory::create_surface(array_gbuffer_normal_and_depth);
	surf_gbuffer_mesh_id_and_triangle_id = CUDAMemory::create_surface(array_gbuffer_mesh_id_and_triangle_id);
	surf_gbuffer_screen_position_prev    = CUDAMemory::create_surface(array_gbuffer_screen_position_prev);

	module.get_global("gbuffer_normal_and_depth")       .set_value(surf_gbuffer_normal_and_depth);
	module.get_global("gbuffer_mesh_id_and_triangle_id").set_value(surf_gbuffer_mesh_id_and_triangle_id);
	module.get_global("gbuffer_screen_position_prev")   .set_value(surf_gbuffer_screen_position_prev);
	
	// Create Frame Buffers
	ptr_frame_buffer_albedo = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_frame_buffer_moment = CUDAMemory::malloc<float4>(screen_pitch * height);

	ptr_direct       = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_indirect     = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_direct_alt   = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_indirect_alt = CUDAMemory::malloc<float4>(screen_pitch * height);
	
	module.get_global("frame_buffer_albedo")  .set_value(ptr_frame_buffer_albedo);
	module.get_global("frame_buffer_moment")  .set_value(ptr_frame_buffer_moment);
	module.get_global("frame_buffer_direct")  .set_value(ptr_direct);
	module.get_global("frame_buffer_indirect").set_value(ptr_indirect);

	// Set Accumulator to a CUDA resource mapping of the GL frame buffer texture
	resource_accumulator = CUDAMemory::resource_register(frame_buffer_handle, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
	surf_accumulator     = CUDAMemory::create_surface(CUDAMemory::resource_get_array(resource_accumulator));
	module.get_global("accumulator").set_value(surf_accumulator);

	// Create History Buffers for SVGF
	ptr_history_length           = CUDAMemory::malloc<int>   (screen_pitch * height);
	ptr_history_direct           = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_history_indirect         = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_history_moment           = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_history_normal_and_depth = CUDAMemory::malloc<float4>(screen_pitch * height);

	module.get_global("history_length")          .set_value(ptr_history_length);
	module.get_global("history_direct")          .set_value(ptr_history_direct);
	module.get_global("history_indirect")        .set_value(ptr_history_indirect);
	module.get_global("history_moment")          .set_value(ptr_history_moment);
	module.get_global("history_normal_and_depth").set_value(ptr_history_normal_and_depth);
	
	// Create Frame Buffers for Temporal Anti-Aliasing
	ptr_taa_frame_prev = CUDAMemory::malloc<float4>(screen_pitch * height);
	ptr_taa_frame_curr = CUDAMemory::malloc<float4>(screen_pitch * height);

	module.get_global("taa_frame_prev").set_value(ptr_taa_frame_prev);
	module.get_global("taa_frame_curr").set_value(ptr_taa_frame_curr);

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
	camera_invalidated = true;

	frames_accumulated = 0;
}

void Pathtracer::resize_free() {
	CUDAMemory::free_array(array_gbuffer_normal_and_depth);
	CUDAMemory::free_array(array_gbuffer_mesh_id_and_triangle_id);
	CUDAMemory::free_array(array_gbuffer_screen_position_prev);

	CUDAMemory::free_surface(surf_gbuffer_normal_and_depth);
	CUDAMemory::free_surface(surf_gbuffer_mesh_id_and_triangle_id);
	CUDAMemory::free_surface(surf_gbuffer_screen_position_prev);

	CUDAMemory::free(ptr_frame_buffer_albedo);
	CUDAMemory::free(ptr_frame_buffer_moment);

	CUDAMemory::resource_unregister(resource_accumulator);
	CUDAMemory::free_surface(surf_accumulator);

	CUDAMemory::free(ptr_direct);
	CUDAMemory::free(ptr_indirect);
	CUDAMemory::free(ptr_direct_alt);
	CUDAMemory::free(ptr_indirect_alt);

	CUDAMemory::free(ptr_history_length);
	CUDAMemory::free(ptr_history_direct);
	CUDAMemory::free(ptr_history_indirect);
	CUDAMemory::free(ptr_history_moment);
	CUDAMemory::free(ptr_history_normal_and_depth);
	
	CUDAMemory::free(ptr_taa_frame_prev);
	CUDAMemory::free(ptr_taa_frame_curr);
}

void Pathtracer::build_tlas() {
	tlas_bvh_builder.build(scene.meshes, scene.mesh_count);

	tlas.index_count = tlas_raw.index_count;
	tlas.indices     = tlas_raw.indices;
	tlas.node_count  = tlas_raw.node_count;

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	const BVH & tlas = tlas_raw;
#else
	tlas_converter.build(tlas_raw);
#endif

	assert(tlas.index_count == scene.mesh_count);
	CUDAMemory::memcpy(ptr_bvh_nodes, tlas.nodes, tlas.node_count);

	int   light_count = 0;
	float light_total_area = 0.0f;

	for (int i = 0; i < scene.mesh_count; i++) {
		const Mesh & mesh = scene.meshes[tlas.indices[i]];

		pinned_mesh_bvh_root_indices[i] = mesh_data_bvh_offsets[mesh.mesh_data_index];

		memcpy(pinned_mesh_transforms     [i].cells, mesh.transform     .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_inv [i].cells, mesh.transform_inv .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_prev[i].cells, mesh.transform_prev.cells, sizeof(Matrix3x4));

		bool is_light = mesh.light_index != -1;
		if (is_light) {
			float light_area_scaled = mesh.light_area * mesh.scale * mesh.scale;

			assert(mesh.light_index < scene.mesh_count);

			pinned_light_mesh_transform_indices[mesh.light_index] = i;
			pinned_light_mesh_area_scaled      [mesh.light_index] = light_area_scaled;
			
			light_count++;
			light_total_area += light_area_scaled;
		}
	}

	CUDAMemory::memcpy(ptr_mesh_bvh_root_indices,  pinned_mesh_bvh_root_indices, scene.mesh_count);
	CUDAMemory::memcpy(ptr_mesh_transforms,        pinned_mesh_transforms,       scene.mesh_count);
	CUDAMemory::memcpy(ptr_mesh_transforms_inv,    pinned_mesh_transforms_inv,   scene.mesh_count);
	CUDAMemory::memcpy(ptr_mesh_transforms_prev,   pinned_mesh_transforms_prev,  scene.mesh_count);
	
	if (scene.has_lights) {
		CUDAMemory::memcpy(ptr_light_total_area, &light_total_area);
		CUDAMemory::memcpy(ptr_light_mesh_transform_indices, pinned_light_mesh_transform_indices, light_count);
		CUDAMemory::memcpy(ptr_light_mesh_area_scaled,       pinned_light_mesh_area_scaled,       light_count);
	}
}

void Pathtracer::update(float delta) {
	if (settings_changed && settings.enable_svgf && settings.camera_aperture > 0.0f) {
		puts("WARNING: SVGF and DoF cannot simultaneously be enabled!");
		settings.camera_aperture = 0.0f;
	}

	if (settings.enable_svgf) {
		scene.camera.update(0.0f, settings);
		scene.update(0.0f);

		build_tlas();

	} else if (settings.enable_scene_update || scene_invalidated) {
		scene.update(delta);

		build_tlas();

		// If SVGF is enabled we can handle Scene updates using reprojection,
		// otherwise 'frames_since_camera_moved' needs to be reset in order to avoid ghosting
		if (!settings.enable_svgf) {
			frames_accumulated = 0;
		}

		scene_invalidated = false;
	}

	if (materials_invalidated) {
		CUDAMemory::memcpy(ptr_materials, scene.materials.data(), scene.materials.size());

		frames_accumulated = 0;
		materials_invalidated = false;
	}

	scene.camera.update(delta, settings);

	if (scene.camera.moved || camera_invalidated) {
		// Upload Camera
		struct CUDACamera {
			Vector3 position;
			Vector3 bottom_left_corner;
			Vector3 x_axis;
			Vector3 y_axis;
			float pixel_spread_angle;
		} cuda_camera;

		cuda_camera.position           = scene.camera.position;
		cuda_camera.bottom_left_corner = scene.camera.bottom_left_corner_rotated;
		cuda_camera.x_axis             = scene.camera.x_axis_rotated;
		cuda_camera.y_axis             = scene.camera.y_axis_rotated;
		cuda_camera.pixel_spread_angle = scene.camera.pixel_spread_angle;

		global_camera.set_value(cuda_camera);

		camera_invalidated = false;
	}

	if (pixel_query_status == PixelQueryStatus::OUTPUT_READY) {
		CUDAMemory::memcpy(&pixel_query_answer, CUDAMemory::Ptr<PixelQueryAnswer>(global_pixel_query_answer.ptr));
		pixel_query_answer.mesh_id = tlas.indices[pixel_query_answer.mesh_id];

		// Reset pixel query
		PixelQuery pixel_query = { -1, -1 };
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

	if (settings_changed) {
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
	CUDAEvent::reset_pool();

	int pixels_left = pixel_count;

	// Render in batches of BATCH_SIZE pixels at a time
	while (pixels_left > 0) {
		int pixel_offset = pixel_count - pixels_left;
		int pixel_count  = Math::min(batch_size, pixels_left);

		CUDAEvent::record(event_info_primary);

		// Generate primary Rays from the current Camera orientation
		kernel_generate.execute(
			Random::get_value(),
			frames_accumulated,
			pixel_offset,
			pixel_count
		);

		for (int bounce = 0; bounce < settings.num_bounces; bounce++) {
			// Extend all Rays that are still alive to their next Triangle intersection
			CUDAEvent::record(event_info_trace[bounce]);

			kernel_trace.execute(bounce);
			
			CUDAEvent::record(event_info_sort[bounce]);
			kernel_sort.execute(Random::get_value(), bounce);

			// Process the various Material types in different Kernels
			//if (scene.has_diffuse) {
				CUDAEvent::record(event_info_shade_diffuse[bounce]);
				kernel_shade_diffuse.execute(Random::get_value(), bounce, frames_accumulated);
			//}

			//if (scene.has_dielectric) {
				CUDAEvent::record(event_info_shade_dielectric[bounce]);
				kernel_shade_dielectric.execute(Random::get_value(), bounce);
			//}

			//if (scene.has_glossy) {
				CUDAEvent::record(event_info_shade_glossy[bounce]);
				kernel_shade_glossy.execute(Random::get_value(), bounce, frames_accumulated);
			//}

			// Trace shadow Rays
			//if (scene.has_lights) {
				CUDAEvent::record(event_info_shadow_trace[bounce]);
				kernel_trace_shadow.execute(bounce);
			//}
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
		CUDAEvent::record(event_info_svgf_reproject);
		kernel_svgf_reproject.execute(frames_accumulated);

		CUdeviceptr direct_in    = ptr_direct    .ptr;
		CUdeviceptr direct_out   = ptr_direct_alt.ptr;
		CUdeviceptr indirect_in  = ptr_indirect    .ptr;
		CUdeviceptr indirect_out = ptr_indirect_alt.ptr;

		if (settings.enable_spatial_variance) {
			// Estimate Variance spatially
			CUDAEvent::record(event_info_svgf_variance);
			kernel_svgf_variance.execute(direct_in, indirect_in, direct_out, indirect_out);

			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);
		}

		// �-Trous Filter
		for (int i = 0; i < settings.atrous_iterations; i++) {
			int step_size = 1 << i;
			
			CUDAEvent::record(event_info_svgf_atrous[i]);
			kernel_svgf_atrous.execute(direct_in, indirect_in, direct_out, indirect_out, step_size);

			// Ping-Pong the Frame Buffers
			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);
		}

		CUDAEvent::record(event_info_svgf_finalize);
		kernel_svgf_finalize.execute(direct_in, indirect_in);

		if (settings.enable_taa) {
			CUDAEvent::record(event_info_taa);

			kernel_taa         .execute(frames_accumulated);
			kernel_taa_finalize.execute();
		}
	} else {
		CUDAEvent::record(event_info_accumulate);
		kernel_accumulate.execute(float(frames_accumulated));
	}

	CUDAEvent::record(event_info_end);
	
	// Reset buffer sizes to default for next frame
	pinned_buffer_sizes->reset(batch_size);
	global_buffer_sizes.set_value(*pinned_buffer_sizes);

	// If a pixel query was previously pending, it has been resolved in the current frame
	if (pixel_query_status == PixelQueryStatus::PENDING) {
		pixel_query_status =  PixelQueryStatus::OUTPUT_READY;
	}
}

void Pathtracer::set_pixel_query(int x, int y) {
	pixel_query.x = x;
	pixel_query.y = screen_height - y; // Y-coordinate is inverted

	CUDAMemory::memcpy(CUDAMemory::Ptr<PixelQuery>(global_pixel_query.ptr), &pixel_query);

	pixel_query_status = PixelQueryStatus::PENDING;
}
