#include "Pathtracer.h"

#include <algorithm>

#include "CUDAContext.h"

#include "MeshData.h"
#include "Material.h"

#include "Random.h"
#include "BlueNoise.h"

#include "Util.h"
#include "ScopeTimer.h"

struct CUDAVector3_SoA {
	CUDAMemory::Ptr<float> x;
	CUDAMemory::Ptr<float> y;
	CUDAMemory::Ptr<float> z;

	inline void init(int buffer_size) {
		x = CUDAMemory::malloc<float>(buffer_size);
		y = CUDAMemory::malloc<float>(buffer_size);
		z = CUDAMemory::malloc<float>(buffer_size);
	}
};

struct TraceBuffer {
	CUDAVector3_SoA origin;
	CUDAVector3_SoA direction;

	CUDAMemory::Ptr<float> cone_width;

	CUDAMemory::Ptr<float>  hit_ts;
	CUDAMemory::Ptr<float4> hits;

	CUDAMemory::Ptr<int> pixel_index;
	CUDAVector3_SoA      throughput;

	CUDAMemory::Ptr<char>  last_material_type;
	CUDAMemory::Ptr<float> last_pdf;

	inline void init(int buffer_size) {
		origin   .init(buffer_size);
		direction.init(buffer_size);

		cone_width = CUDAMemory::malloc<float>(buffer_size);

		hit_ts = CUDAMemory::malloc<float> (buffer_size);
		hits   = CUDAMemory::malloc<float4>(buffer_size);

		pixel_index = CUDAMemory::malloc<int>(buffer_size);
		throughput.init(buffer_size);

		last_material_type = CUDAMemory::malloc<char> (buffer_size);
		last_pdf           = CUDAMemory::malloc<float>(buffer_size);
	}
};

struct MaterialBuffer {
	CUDAVector3_SoA direction;
	
	CUDAMemory::Ptr<float> cone_width;

	CUDAMemory::Ptr<float>  hit_ts;
	CUDAMemory::Ptr<float4> hits;

	CUDAMemory::Ptr<int> pixel_index;
	CUDAVector3_SoA      throughput;

	inline void init(int buffer_size) {
		direction.init(buffer_size);
		
		cone_width = CUDAMemory::malloc<float>(buffer_size);

		hit_ts = CUDAMemory::malloc<float> (buffer_size);
		hits   = CUDAMemory::malloc<float4>(buffer_size);

		pixel_index  = CUDAMemory::malloc<int>(buffer_size);
		throughput.init(buffer_size);
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
};

struct BufferSizes {
	int trace     [NUM_BOUNCES];
	int diffuse   [NUM_BOUNCES];
	int dielectric[NUM_BOUNCES];
	int glossy    [NUM_BOUNCES];
	int shadow    [NUM_BOUNCES];

	int rays_retired       [NUM_BOUNCES];
	int rays_retired_shadow[NUM_BOUNCES];
};
static BufferSizes * buffer_sizes; // Pinned memory (Non-Pageable)

static void upload_camera(CUDAModule::Global global_camera, const Camera & camera) {
	struct CUDACamera {
		Vector3 position;
		Vector3 bottom_left_corner;
		Vector3 x_axis;
		Vector3 y_axis;
	} cuda_camera;

	cuda_camera.position           = camera.position;
	cuda_camera.bottom_left_corner = camera.bottom_left_corner_rotated;
	cuda_camera.x_axis             = camera.x_axis_rotated;
	cuda_camera.y_axis             = camera.y_axis_rotated;

	global_camera.set_value(cuda_camera);
}

void Pathtracer::init(int mesh_count, char const ** mesh_names, char const * sky_name, unsigned frame_buffer_handle) {
	ScopeTimer timer("Pathtracer Initialization");

	pixel_count = SCREEN_WIDTH * SCREEN_HEIGHT;
	batch_size  = BATCH_SIZE;

	CUDAContext::init();

	scene.init(mesh_count, mesh_names, sky_name);

	// Init CUDA Module and its Kernel
	module.init("CUDA_Source/Pathtracer.cu", CUDAContext::compute_capability, MAX_REGISTERS);

	// Set global Material table
	module.get_global("materials").set_buffer(Material::materials);

	// Set global Texture table
	int texture_count = Texture::textures.size();
	if (texture_count > 0) {
		CUtexObject * tex_objects = new CUtexObject[texture_count];
		
		// Get maximum anisotropy from OpenGL
		int max_aniso; glGetIntegerv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_aniso);

		for (int i = 0; i < texture_count; i++) {
			const Texture & texture = Texture::textures[i];

			// Create mipmapped CUDA array
			CUmipmappedArray array = CUDAMemory::create_array_mipmap(
				texture.width,
				texture.height,
				texture.channels,
				texture.get_cuda_array_format(),
				texture.mip_levels
			);

			// Upload each level of the mipmap
			for (int level = 0; level < texture.mip_levels; level++) {
				CUarray level_array;
				CUDACALL(cuMipmappedArrayGetLevel(&level_array, array, level));

				int level_width_in_bytes = texture.get_width_in_bytes() >> level;
				int level_height         = texture.height               >> level;

				CUDAMemory::copy_array(level_array, level_width_in_bytes, level_height, texture.data + texture.mip_offsets[level]);
			}

			// Describe the Array to read from
			CUDA_RESOURCE_DESC res_desc = { };
			res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_MIPMAPPED_ARRAY;
			res_desc.res.mipmap.hMipmappedArray = array;

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
			tex_desc.maxMipmapLevelClamp = texture.mip_levels;
			tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;

			// Describe the Texture View
			CUDA_RESOURCE_VIEW_DESC view_desc = { };
			view_desc.format = texture.get_cuda_resource_view_format();
			view_desc.width  = texture.get_cuda_resource_view_width();
			view_desc.height = texture.get_cuda_resource_view_height();
			view_desc.firstMipmapLevel = 0;
			view_desc.lastMipmapLevel  = texture.mip_levels;

			CUDACALL(cuTexObjectCreate(tex_objects + i, &res_desc, &tex_desc, &view_desc));
		}

		module.get_global("textures").set_buffer(tex_objects, texture_count);

		delete [] tex_objects;
	}

	int mesh_data_count = MeshData::mesh_datas.size();

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

		global_bvh_node_count += MeshData::mesh_datas[i]->bvh.node_count;
		global_index_count    += MeshData::mesh_datas[i]->bvh.index_count;
		global_triangle_count += MeshData::mesh_datas[i]->triangle_count;
	}

	BVHNodeType * global_bvh_nodes = new BVHNodeType[global_bvh_node_count];
	int         * global_indices   = new int        [global_index_count];
	Triangle    * global_triangles = new Triangle   [global_triangle_count];

	for (int m = 0; m < mesh_data_count; m++) {
		const MeshData * mesh_data = MeshData::mesh_datas[m];

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
	pinned_light_mesh_transform_indices = CUDAMemory::malloc_pinned<int>      (scene.mesh_count);
	pinned_light_mesh_area_scaled       = CUDAMemory::malloc_pinned<float>    (scene.mesh_count);

	ptr_mesh_bvh_root_indices = CUDAMemory::malloc<int>      (scene.mesh_count);
	ptr_mesh_transforms       = CUDAMemory::malloc<Matrix3x4>(scene.mesh_count);
	ptr_mesh_transforms_inv   = CUDAMemory::malloc<Matrix3x4>(scene.mesh_count);

	module.get_global("mesh_bvh_root_indices").set_value(ptr_mesh_bvh_root_indices);
	module.get_global("mesh_transforms")      .set_value(ptr_mesh_transforms);
	module.get_global("mesh_transforms_inv")  .set_value(ptr_mesh_transforms_inv);
	
	ptr_bvh_nodes = CUDAMemory::malloc<BVHNodeType>(global_bvh_node_count);
	CUDAMemory::memcpy(ptr_bvh_nodes, global_bvh_nodes, global_bvh_node_count);

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	module.get_global("bvh_nodes").set_value(ptr_bvh_nodes);
#elif BVH_TYPE == BVH_QBVH
	module.get_global("qbvh_nodes").set_value(ptr_bvh_nodes);
#elif BVH_TYPE == BVH_CWBVH
	module.get_global("cwbvh_nodes").set_value(ptr_bvh_nodes);
#endif

	tlas_bvh_builder.init(&tlas_raw, mesh_count, 1);

	tlas_raw.node_count = mesh_count * 2;
#if BVH_TYPE == BVH_QBVH || BVH_TYPE == BVH_CWBVH
	tlas_converter.init(&tlas, tlas_raw);
#endif

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
	
	CUDATriangle * triangles             = new CUDATriangle[global_index_count];
	int          * triangle_material_ids = new int         [global_index_count];

	int * reverse_indices = new int[global_index_count];

	for (int i = 0; i < global_index_count; i++) {
		int index = global_indices[i];

		triangles[i].position_0      = global_triangles[index].position_0;
		triangles[i].position_edge_1 = global_triangles[index].position_1 - global_triangles[index].position_0;
		triangles[i].position_edge_2 = global_triangles[index].position_2 - global_triangles[index].position_0;

		triangles[i].normal_0      = global_triangles[index].normal_0;
		triangles[i].normal_edge_1 = global_triangles[index].normal_1 - global_triangles[index].normal_0;
		triangles[i].normal_edge_2 = global_triangles[index].normal_2 - global_triangles[index].normal_0;

		triangles[i].tex_coord_0      = global_triangles[index].tex_coord_0;
		triangles[i].tex_coord_edge_1 = global_triangles[index].tex_coord_1 - global_triangles[index].tex_coord_0;
		triangles[i].tex_coord_edge_2 = global_triangles[index].tex_coord_2 - global_triangles[index].tex_coord_0;

		triangle_material_ids[i] = global_triangles[index].material_id;

		reverse_indices[index] = i;
	}

	module.get_global("triangles")            .set_buffer(triangles,             global_index_count);
	module.get_global("triangle_material_ids").set_buffer(triangle_material_ids, global_index_count);

	// Init OpenGL MeshData for rasterization
	for (int m = 0; m < mesh_data_count; m++) {
		MeshData::mesh_datas[m]->gl_init(reverse_indices + mesh_data_triangle_offsets[m]);
	}

	// Initialize OpenGL Shaders
	shader = Shader::load(
		DATA_PATH("Shaders/primary_vertex.glsl"),
		DATA_PATH("Shaders/primary_fragment.glsl")
	);
	shader.bind();

	uniform_jitter               = shader.get_uniform("jitter");
	uniform_view_projection      = shader.get_uniform("view_projection");
	uniform_view_projection_prev = shader.get_uniform("view_projection_prev");

	uniform_transform      = shader.get_uniform("transform");
	uniform_transform_prev = shader.get_uniform("transform_prev");

	uniform_mesh_id = shader.get_uniform("mesh_id");

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
			const MeshData * mesh_data = MeshData::mesh_datas[m];

			LightMesh * light_mesh = nullptr;

			// For every Triangle, check whether it is a Light based on its Material
			for (int t = 0; t < mesh_data->triangle_count; t++) {
				const Triangle & triangle = mesh_data->triangles[t];

				if (Material::materials[mesh_data->material_offset + triangle.material_id].type == Material::Type::LIGHT) {
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

		module.get_global("light_indices")         .set_buffer(light_indices,          light_triangles.size());
		module.get_global("light_areas_cumulative").set_buffer(light_areas_cumulative, light_triangles.size());

		delete [] light_indices;
		delete [] light_areas_cumulative;

		float * light_mesh_area_unscaled        = MALLOCA(float, mesh_count);
		int   * light_mesh_triangle_count       = MALLOCA(int,   mesh_count);
		int   * light_mesh_triangle_first_index = MALLOCA(int,   mesh_count);
		
		int light_total_count = 0;
		int light_mesh_count  = 0;
		
		for (int m = 0; m < mesh_count; m++) {
			int light_mesh_data_index = light_mesh_data_indices[scene.meshes[m].mesh_data_index];

			if (light_mesh_data_index != -1) {
				const LightMesh & light_mesh = light_meshes[light_mesh_data_index];

				scene.meshes[m].light_index = light_mesh_count;
				scene.meshes[m].light_area  = light_mesh.area;

				int mesh_index = light_mesh_count++;
				assert(mesh_index < mesh_count);

				light_mesh_area_unscaled       [mesh_index] = light_mesh.area;
				light_mesh_triangle_first_index[mesh_index] = light_mesh.triangle_first_index;
				light_mesh_triangle_count      [mesh_index] = light_mesh.triangle_count;

				light_total_count += light_mesh.triangle_count;
			}
		}
		
		module.get_global("light_total_count_inv").set_value(1.0f / float(light_total_count));
		module.get_global("light_mesh_count")     .set_value(light_mesh_count);

		module.get_global("light_mesh_area_unscaled")       .set_buffer(light_mesh_area_unscaled,        light_mesh_count);
		module.get_global("light_mesh_triangle_count")      .set_buffer(light_mesh_triangle_count,       light_mesh_count);
		module.get_global("light_mesh_triangle_first_index").set_buffer(light_mesh_triangle_first_index, light_mesh_count);

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
	
	delete [] triangles;
	delete [] reverse_indices;

	module.get_global("sky_size").set_value (scene.sky.size);
	module.get_global("sky_data").set_buffer(scene.sky.data, scene.sky.size * scene.sky.size);
	
	// Set Blue Noise Sampler globals
	module.get_global("sobol_256spp_256d").set_buffer(sobol_256spp_256d);
	module.get_global("scrambling_tile").set_buffer(scrambling_tile);
	module.get_global("ranking_tile").set_buffer(ranking_tile);

	// Initialize buffers used by Wavefront kernels
	TraceBuffer     ray_buffer_trace;                                      ray_buffer_trace           .init(batch_size);
	MaterialBuffer  ray_buffer_shade_diffuse;    if (scene.has_diffuse)    ray_buffer_shade_diffuse   .init(batch_size);
	MaterialBuffer  ray_buffer_shade_dielectric; if (scene.has_dielectric) ray_buffer_shade_dielectric.init(batch_size);
	MaterialBuffer  ray_buffer_shade_glossy;     if (scene.has_glossy)     ray_buffer_shade_glossy    .init(batch_size);
	ShadowRayBuffer ray_buffer_shadow;           if (scene.has_lights)     ray_buffer_shadow          .init(batch_size);

	module.get_global("ray_buffer_trace")           .set_value(ray_buffer_trace);
	module.get_global("ray_buffer_shade_diffuse")   .set_value(ray_buffer_shade_diffuse);
	module.get_global("ray_buffer_shade_dielectric").set_value(ray_buffer_shade_dielectric);
	module.get_global("ray_buffer_shade_glossy")    .set_value(ray_buffer_shade_glossy);
	module.get_global("ray_buffer_shadow")          .set_value(ray_buffer_shadow);

	buffer_sizes = CUDAMemory::malloc_pinned<BufferSizes>();
	memset(buffer_sizes, 0, sizeof(BufferSizes));
	buffer_sizes->trace[0] = batch_size;

	global_buffer_sizes = module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(*buffer_sizes);

	global_settings = module.get_global("settings");

	unsigned long long bytes_available = CUDAContext::get_available_memory();
	unsigned long long bytes_allocated = CUDAContext::total_memory - bytes_available;

	puts("");
	printf("CUDA Memory allocated: %8llu KB (%6llu MB)\n", bytes_allocated >> 10, bytes_allocated >> 20);
	printf("CUDA Memory free:      %8llu KB (%6llu MB)\n", bytes_available >> 10, bytes_available >> 20);

	kernel_primary         .init(&module, "kernel_primary");
	kernel_generate        .init(&module, "kernel_generate");
	kernel_trace           .init(&module, "kernel_trace");
	kernel_sort            .init(&module, "kernel_sort");
	kernel_shade_diffuse   .init(&module, "kernel_shade_diffuse");
	kernel_shade_dielectric.init(&module, "kernel_shade_dielectric");
	kernel_shade_glossy    .init(&module, "kernel_shade_glossy");
	kernel_trace_shadow    .init(&module, "kernel_trace_shadow");
	kernel_svgf_temporal   .init(&module, "kernel_svgf_temporal");
	kernel_svgf_variance   .init(&module, "kernel_svgf_variance");
	kernel_svgf_atrous     .init(&module, "kernel_svgf_atrous");
	kernel_svgf_finalize   .init(&module, "kernel_svgf_finalize");
	kernel_taa             .init(&module, "kernel_taa");
	kernel_taa_finalize    .init(&module, "kernel_taa_finalize");
	kernel_reconstruct     .init(&module, "kernel_reconstruct");
	kernel_accumulate      .init(&module, "kernel_accumulate");

	// Set Block dimensions for all Kernels
	kernel_svgf_temporal.occupancy_max_block_size_2d();
	kernel_svgf_variance.occupancy_max_block_size_2d();
	kernel_svgf_atrous  .occupancy_max_block_size_2d();
	kernel_svgf_finalize.occupancy_max_block_size_2d();
	kernel_taa          .occupancy_max_block_size_2d();
	kernel_taa_finalize .occupancy_max_block_size_2d();
	kernel_reconstruct  .occupancy_max_block_size_2d();
	kernel_accumulate   .occupancy_max_block_size_2d();

	kernel_primary         .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_generate        .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_sort            .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_diffuse   .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_dielectric.set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_glossy    .set_block_dim(WARP_SIZE * 2, 1, 1);
	
#if BVH_TYPE == BVH_CWBVH
	static constexpr int bvh_stack_element_size = 8; // CWBVH uses a stack of int2's (8 bytes)
#else
	static constexpr int bvh_stack_element_size = 4; // Other BVH's use a stack of ints (4 bytes)
#endif

	CUoccupancyB2DSize block_size_to_shared_memory = [](int block_size) {
		return size_t(block_size) * SHARED_STACK_SIZE * bvh_stack_element_size;
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

	// Initialize timers
	event_primary.init("Primary", "Primary");

	for (int i = 0; i < NUM_BOUNCES; i++) {
		const int len = 16;
		char    * category = new char[len];
		sprintf_s(category, len, "Bounce %i", i);

		event_trace           [i].init(category, "Trace");
		event_sort            [i].init(category, "Sort");
		event_shade_diffuse   [i].init(category, "Diffuse");
		event_shade_dielectric[i].init(category, "Dielectric");
		event_shade_glossy    [i].init(category, "Glossy");
		event_shadow_trace    [i].init(category, "Shadow");
	}

	event_svgf_temporal.init("SVGF", "Temporal");
	event_svgf_variance.init("SVGF", "Variance");
	for (int i = 0; i < MAX_ATROUS_ITERATIONS; i++) {
		const int len = 16;
		char    * name = new char[len];
		sprintf_s(name, len, "A Trous %i", i);

		event_svgf_atrous[i].init("SVGF", name);
	}
	event_svgf_finalize.init("SVGF", "Finalize");

	event_taa        .init("Post", "TAA");
	event_reconstruct.init("Post", "Reconstruct");
	event_accumulate .init("Post", "Accumulate");

	event_end.init("END", "END");

	resize_init(frame_buffer_handle, SCREEN_WIDTH, SCREEN_HEIGHT);
	
	// Realloc as pinned memory
	delete [] tlas.nodes;
	tlas.nodes = CUDAMemory::malloc_pinned<BVHNodeType>(2 * mesh_count);

	scene.update(0.0f);
	build_tlas();
}

void Pathtracer::resize_init(unsigned frame_buffer_handle, int width, int height) {
	pixel_count = width * height;
	batch_size  = Math::min(BATCH_SIZE, pixel_count);

	int pitch = Math::divide_round_up(width, WARP_SIZE) * WARP_SIZE;

	module.get_global("screen_width") .set_value(width);
	module.get_global("screen_pitch") .set_value(pitch);
	module.get_global("screen_height").set_value(height);

	// Resize GBuffers
	gbuffer.resize(width, height);

	resource_gbuffer_normal_and_depth = CUDAMemory::resource_register(gbuffer.buffer_normal_and_depth,        CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
	resource_gbuffer_uv               = CUDAMemory::resource_register(gbuffer.buffer_uv,                      CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
	resource_gbuffer_uv_gradient      = CUDAMemory::resource_register(gbuffer.buffer_uv_gradient,             CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
	resource_gbuffer_triangle_id      = CUDAMemory::resource_register(gbuffer.buffer_mesh_id_and_triangle_id, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
	resource_gbuffer_motion     	  = CUDAMemory::resource_register(gbuffer.buffer_motion,                  CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
	resource_gbuffer_z_gradient    	  = CUDAMemory::resource_register(gbuffer.buffer_z_gradient,              CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);

	module.set_texture("gbuffer_normal_and_depth",        CUDAMemory::resource_get_array(resource_gbuffer_normal_and_depth), CU_TR_FILTER_MODE_POINT);
	module.set_texture("gbuffer_uv",                      CUDAMemory::resource_get_array(resource_gbuffer_uv),               CU_TR_FILTER_MODE_POINT);
	module.set_texture("gbuffer_uv_gradient",             CUDAMemory::resource_get_array(resource_gbuffer_uv_gradient),      CU_TR_FILTER_MODE_POINT);
	module.set_texture("gbuffer_mesh_id_and_triangle_id", CUDAMemory::resource_get_array(resource_gbuffer_triangle_id),      CU_TR_FILTER_MODE_POINT);
	module.set_texture("gbuffer_screen_position_prev",    CUDAMemory::resource_get_array(resource_gbuffer_motion),           CU_TR_FILTER_MODE_POINT);
	module.set_texture("gbuffer_depth_gradient",          CUDAMemory::resource_get_array(resource_gbuffer_z_gradient),       CU_TR_FILTER_MODE_POINT);

	// Create Frame Buffers
	module.get_global("frame_buffer_albedo").set_value(CUDAMemory::malloc<float4>(pitch * height).ptr);
	module.get_global("frame_buffer_moment").set_value(CUDAMemory::malloc<float4>(pitch * height).ptr);
	
	ptr_direct       = CUDAMemory::malloc<float4>(pitch * height);
	ptr_indirect     = CUDAMemory::malloc<float4>(pitch * height);
	ptr_direct_alt   = CUDAMemory::malloc<float4>(pitch * height);
	ptr_indirect_alt = CUDAMemory::malloc<float4>(pitch * height);

	module.get_global("frame_buffer_direct")  .set_value(ptr_direct  .ptr);
	module.get_global("frame_buffer_indirect").set_value(ptr_indirect.ptr);

	module.get_global("sample_xy")     .set_value(CUDAMemory::malloc<float2>(pitch * height).ptr);
	module.get_global("reconstruction").set_value(CUDAMemory::malloc<float4>(pitch * height).ptr);

	// Set Accumulator to a CUDA resource mapping of the GL frame buffer texture
	resource_accumulator = CUDAMemory::resource_register(frame_buffer_handle, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
	module.set_surface("accumulator", CUDAMemory::resource_get_array(resource_accumulator));

	// Create History Buffers for SVGF
	module.get_global("history_length")          .set_value(CUDAMemory::malloc<int>   (pitch * height).ptr);
	module.get_global("history_direct")          .set_value(CUDAMemory::malloc<float4>(pitch * height).ptr);
	module.get_global("history_indirect")        .set_value(CUDAMemory::malloc<float4>(pitch * height).ptr);
	module.get_global("history_moment")          .set_value(CUDAMemory::malloc<float4>(pitch * height).ptr);
	module.get_global("history_normal_and_depth").set_value(CUDAMemory::malloc<float4>(pitch * height).ptr);
	
	// Create Frame Buffers for Temporal Anti-Aliasing
	module.get_global("taa_frame_prev").set_value(CUDAMemory::malloc<float4>(pitch * height));
	module.get_global("taa_frame_curr").set_value(CUDAMemory::malloc<float4>(pitch * height));

	// Set Grid dimensions for screen size dependent Kernels
	kernel_svgf_temporal.set_grid_dim(pitch / kernel_svgf_temporal.block_dim_x, Math::divide_round_up(height, kernel_svgf_temporal.block_dim_y), 1);
	kernel_svgf_variance.set_grid_dim(pitch / kernel_svgf_variance.block_dim_x, Math::divide_round_up(height, kernel_svgf_variance.block_dim_y), 1);
	kernel_svgf_atrous  .set_grid_dim(pitch / kernel_svgf_atrous  .block_dim_x, Math::divide_round_up(height, kernel_svgf_atrous  .block_dim_y), 1);
	kernel_svgf_finalize.set_grid_dim(pitch / kernel_svgf_finalize.block_dim_x, Math::divide_round_up(height, kernel_svgf_finalize.block_dim_y), 1);
	kernel_taa          .set_grid_dim(pitch / kernel_taa          .block_dim_x, Math::divide_round_up(height, kernel_taa          .block_dim_y), 1);
	kernel_taa_finalize .set_grid_dim(pitch / kernel_taa_finalize .block_dim_x, Math::divide_round_up(height, kernel_taa_finalize .block_dim_y), 1);
	kernel_reconstruct  .set_grid_dim(pitch / kernel_reconstruct  .block_dim_x, Math::divide_round_up(height, kernel_reconstruct  .block_dim_y), 1);
	kernel_accumulate   .set_grid_dim(pitch / kernel_accumulate   .block_dim_x, Math::divide_round_up(height, kernel_accumulate   .block_dim_y), 1);

	kernel_primary         .set_grid_dim(Math::divide_round_up(batch_size, kernel_primary         .block_dim_x), 1, 1);
	kernel_generate        .set_grid_dim(Math::divide_round_up(batch_size, kernel_generate        .block_dim_x), 1, 1);
	kernel_sort            .set_grid_dim(Math::divide_round_up(batch_size, kernel_sort            .block_dim_x), 1, 1);
	kernel_shade_diffuse   .set_grid_dim(Math::divide_round_up(batch_size, kernel_shade_diffuse   .block_dim_x), 1, 1);
	kernel_shade_dielectric.set_grid_dim(Math::divide_round_up(batch_size, kernel_shade_dielectric.block_dim_x), 1, 1);
	kernel_shade_glossy    .set_grid_dim(Math::divide_round_up(batch_size, kernel_shade_glossy    .block_dim_x), 1, 1);
	
	scene.camera.resize(width, height);
	frames_accumulated = 0;
	
	scene.camera.update(0.0f, settings.enable_rasterization);

	global_camera = module.get_global("camera");
	upload_camera(global_camera, scene.camera);
}

void Pathtracer::resize_free() {
	CUDAMemory::resource_unregister(resource_gbuffer_normal_and_depth);
	CUDAMemory::resource_unregister(resource_gbuffer_uv);
	CUDAMemory::resource_unregister(resource_gbuffer_uv_gradient);
	CUDAMemory::resource_unregister(resource_gbuffer_triangle_id);
	CUDAMemory::resource_unregister(resource_gbuffer_motion);
	CUDAMemory::resource_unregister(resource_gbuffer_z_gradient);

	CUDACALL(cuTexObjectDestroy(module.get_global("gbuffer_normal_and_depth")	    .get_value<CUtexObject>()));
	CUDACALL(cuTexObjectDestroy(module.get_global("gbuffer_uv")					    .get_value<CUtexObject>()));
	CUDACALL(cuTexObjectDestroy(module.get_global("gbuffer_uv_gradient")		    .get_value<CUtexObject>()));
	CUDACALL(cuTexObjectDestroy(module.get_global("gbuffer_mesh_id_and_triangle_id").get_value<CUtexObject>()));
	CUDACALL(cuTexObjectDestroy(module.get_global("gbuffer_screen_position_prev")   .get_value<CUtexObject>()));
	CUDACALL(cuTexObjectDestroy(module.get_global("gbuffer_depth_gradient")         .get_value<CUtexObject>()));
	
	CUDAMemory::free(module.get_global("frame_buffer_albedo").get_value<CUDAMemory::Ptr<float4>>());
	CUDAMemory::free(module.get_global("frame_buffer_moment").get_value<CUDAMemory::Ptr<float4>>());

	CUDAMemory::free(module.get_global("sample_xy")     .get_value<CUDAMemory::Ptr<float2>>());
	CUDAMemory::free(module.get_global("reconstruction").get_value<CUDAMemory::Ptr<float4>>());
	
	CUDAMemory::resource_unregister(resource_accumulator);
	CUDACALL(cuSurfObjectDestroy(module.get_global("accumulator").get_value<CUsurfObject>()));

	CUDAMemory::free(ptr_direct);
	CUDAMemory::free(ptr_indirect);
	CUDAMemory::free(ptr_direct_alt);
	CUDAMemory::free(ptr_indirect_alt);

	CUDAMemory::free(module.get_global("history_length")          .get_value<CUDAMemory::Ptr<int>>   ());
	CUDAMemory::free(module.get_global("history_direct")          .get_value<CUDAMemory::Ptr<float4>>());
	CUDAMemory::free(module.get_global("history_indirect")        .get_value<CUDAMemory::Ptr<float4>>());
	CUDAMemory::free(module.get_global("history_moment")          .get_value<CUDAMemory::Ptr<float4>>());
	CUDAMemory::free(module.get_global("history_normal_and_depth").get_value<CUDAMemory::Ptr<float4>>());
	
	CUDAMemory::free(module.get_global("taa_frame_prev").get_value<CUDAMemory::Ptr<float4>>());
	CUDAMemory::free(module.get_global("taa_frame_curr").get_value<CUDAMemory::Ptr<float4>>());
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

	CUDAMemory::memcpy<BVHNodeType>(ptr_bvh_nodes, tlas.nodes, tlas.node_count);

	assert(tlas.index_count == scene.mesh_count);

	int   light_count = 0;
	float light_total_area = 0.0f;

	for (int i = 0; i < scene.mesh_count; i++) {
		const Mesh & mesh = scene.meshes[tlas.indices[i]];

		pinned_mesh_bvh_root_indices[i] = mesh_data_bvh_offsets[mesh.mesh_data_index];

		memcpy(pinned_mesh_transforms    [i].cells, mesh.transform    .cells, sizeof(Matrix3x4));
		memcpy(pinned_mesh_transforms_inv[i].cells, mesh.transform_inv.cells, sizeof(Matrix3x4));

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

	CUDAMemory::memcpy(ptr_mesh_bvh_root_indices, pinned_mesh_bvh_root_indices, scene.mesh_count);
	CUDAMemory::memcpy(ptr_mesh_transforms,       pinned_mesh_transforms,       scene.mesh_count);
	CUDAMemory::memcpy(ptr_mesh_transforms_inv,   pinned_mesh_transforms_inv,   scene.mesh_count);
	
	if (scene.has_lights) {
		CUDAMemory::memcpy(ptr_light_total_area, &light_total_area);
		CUDAMemory::memcpy(ptr_light_mesh_transform_indices, pinned_light_mesh_transform_indices, light_count);
		CUDAMemory::memcpy(ptr_light_mesh_area_scaled,       pinned_light_mesh_area_scaled,       light_count);
	}
}

void Pathtracer::update(float delta) {
	if (settings.enable_scene_update) {
		scene.update(delta);

		build_tlas();

		// If SVGF is enabled we can handle Scene updates using reprojection,
		// otherwise 'frames_since_camera_moved' needs to be reset in order to avoid ghosting
		if (!settings.enable_svgf) {
			frames_accumulated = 0;
		}
	} else {
		scene.update(0.0f); // Update with 0 delta to make sure previous Transforms match current Transforms
	}

	scene.camera.update(delta, settings.enable_rasterization);

	if (scene.camera.moved) {
		upload_camera(global_camera, scene.camera);
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

#define RECORD_EVENT(e) (e.record(), events.push_back(&e))

void Pathtracer::render() {
	events.clear();

	if (settings.enable_rasterization) {
		gbuffer.bind();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shader.bind();

		glUniform2f(uniform_jitter, scene.camera.jitter.x, scene.camera.jitter.y);

		glUniformMatrix4fv(uniform_view_projection,      1, GL_TRUE, reinterpret_cast<const GLfloat *>(&scene.camera.view_projection));
		glUniformMatrix4fv(uniform_view_projection_prev, 1, GL_TRUE, reinterpret_cast<const GLfloat *>(&scene.camera.view_projection_prev));

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(3);
		
		for (int m = 0; m < scene.mesh_count; m++) {
			const Mesh & mesh = scene.meshes[tlas.indices[m]];
			
			glUniformMatrix4fv(uniform_transform,      1, GL_TRUE, reinterpret_cast<const GLfloat *>(&mesh.transform));
			glUniformMatrix4fv(uniform_transform_prev, 1, GL_TRUE, reinterpret_cast<const GLfloat *>(&mesh.transform_prev));

			glUniform1i(uniform_mesh_id, m);

			MeshData::mesh_datas[mesh.mesh_data_index]->gl_render();
		}

		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(2);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);

		shader .unbind();
		gbuffer.unbind();

		glFinish();
	}

	int pixels_left = pixel_count;

	// Render in batches of BATCH_SIZE pixels at a time
	while (pixels_left > 0) {
		int pixel_offset = pixel_count - pixels_left;
		int pixel_count  = pixels_left > batch_size ? batch_size : pixels_left;

		RECORD_EVENT(event_primary);

		if (settings.enable_rasterization) {
			// Convert rasterized GBuffers into primary Rays
			kernel_primary.execute(
				Random::get_value(),
				frames_accumulated,
				pixel_offset,
				pixel_count,
				settings.enable_taa
			);
		} else {
			// Generate primary Rays from the current Camera orientation
			kernel_generate.execute(
				Random::get_value(),
				frames_accumulated,
				pixel_offset,
				pixel_count
			);
		}

		for (int bounce = 0; bounce < NUM_BOUNCES; bounce++) {
			// When rasterizing primary rays we can skip tracing rays on bounce 0
			if (!(bounce == 0 && settings.enable_rasterization)) {
				// Extend all Rays that are still alive to their next Triangle intersection
				RECORD_EVENT(event_trace[bounce]);
				kernel_trace.execute(bounce);
			
				RECORD_EVENT(event_sort[bounce]);
				kernel_sort.execute(Random::get_value(), bounce);
			}

			// Process the various Material types in different Kernels
			if (scene.has_diffuse) {
				RECORD_EVENT(event_shade_diffuse[bounce]);
				kernel_shade_diffuse.execute(Random::get_value(), bounce, frames_accumulated);
			}

			if (scene.has_dielectric) {
				RECORD_EVENT(event_shade_dielectric[bounce]);
				kernel_shade_dielectric.execute(Random::get_value(), bounce);
			}

			if (scene.has_glossy) {
				RECORD_EVENT(event_shade_glossy[bounce]);
				kernel_shade_glossy.execute(Random::get_value(), bounce, frames_accumulated);
			}

			// Trace shadow Rays
			if (scene.has_lights) {
				RECORD_EVENT(event_shadow_trace[bounce]);
				kernel_trace_shadow.execute(bounce);
			}
		}

		pixels_left -= batch_size;

		if (pixels_left > 0) {
			// Set buffer sizes to appropriate pixel count for next Batch
			buffer_sizes->trace[0] = Math::min(batch_size, pixels_left);
			global_buffer_sizes.set_value(*buffer_sizes);
		}
	}

	if (settings.enable_svgf) {
		// Integrate temporally
		RECORD_EVENT(event_svgf_temporal);
		kernel_svgf_temporal.execute();

		CUdeviceptr direct_in    = ptr_direct    .ptr;
		CUdeviceptr direct_out   = ptr_direct_alt.ptr;
		CUdeviceptr indirect_in  = ptr_indirect    .ptr;
		CUdeviceptr indirect_out = ptr_indirect_alt.ptr;

		if (settings.enable_spatial_variance) {
			// Estimate Variance spatially
			RECORD_EVENT(event_svgf_variance);
			kernel_svgf_variance.execute(direct_in, indirect_in, direct_out, indirect_out);
		} else {
			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);
		}

		// À-Trous Filter
		for (int i = 0; i < settings.atrous_iterations; i++) {
			int step_size = 1 << i;
				
			// Ping-Pong the Frame Buffers
			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);

			RECORD_EVENT(event_svgf_atrous[i]);
			kernel_svgf_atrous.execute(direct_in, indirect_in, direct_out, indirect_out, step_size);
		}

		RECORD_EVENT(event_svgf_finalize);
		kernel_svgf_finalize.execute(direct_out, indirect_out);

		if (settings.enable_taa) {
			RECORD_EVENT(event_taa);

			kernel_taa         .execute();
			kernel_taa_finalize.execute();
		}
	} else {
		if (settings.reconstruction_filter != ReconstructionFilter::BOX) {
			RECORD_EVENT(event_reconstruct);
			kernel_reconstruct.execute();
		}

		RECORD_EVENT(event_accumulate);
		kernel_accumulate.execute(float(frames_accumulated));
	}

	RECORD_EVENT(event_end);
	
	// Reset buffer sizes to default for next frame
	buffer_sizes->trace[0] = batch_size;
	global_buffer_sizes.set_value(*buffer_sizes);
}
