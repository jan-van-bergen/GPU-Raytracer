#include "Pathtracer.h"

#include <SDL2/SDL.h>

#include "CUDAContext.h"

#include "MeshData.h"
#include "BVH.h"
#include "QBVH.h"
#include "CWBVH.h"

#include "Sky.h"

#include "BlueNoise.h"

#include "ScopedTimer.h"

struct Vertex {
	Vector3 position;
	Vector3 normal;
	Vector2 uv;
	int     triangle_id;
};

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
		
	CUDAMemory::Ptr<int> triangle_id;
	CUDAMemory::Ptr<float> u;
	CUDAMemory::Ptr<float> v;

	CUDAMemory::Ptr<int> pixel_index;
	CUDAVector3_SoA      throughput;

	CUDAMemory::Ptr<char>  last_material_type;
	CUDAMemory::Ptr<float> last_pdf;

	inline void init(int buffer_size) {
		origin   .init(buffer_size);
		direction.init(buffer_size);
		
		triangle_id = CUDAMemory::malloc<int>  (buffer_size);
		u           = CUDAMemory::malloc<float>(buffer_size);
		v           = CUDAMemory::malloc<float>(buffer_size);

		pixel_index = CUDAMemory::malloc<int>(buffer_size);
		throughput.init(buffer_size);

		last_material_type = CUDAMemory::malloc<char> (buffer_size);
		last_pdf           = CUDAMemory::malloc<float>(buffer_size);
	}
};

struct MaterialBuffer {
	CUDAVector3_SoA direction;
	
	CUDAMemory::Ptr<int> triangle_id;
	CUDAMemory::Ptr<float> u;
	CUDAMemory::Ptr<float> v;

	CUDAMemory::Ptr<int> pixel_index;
	CUDAVector3_SoA      throughput;

	inline void init(int buffer_size) {
		direction.init(buffer_size);

		triangle_id = CUDAMemory::malloc<int>  (buffer_size);
		u           = CUDAMemory::malloc<float>(buffer_size);
		v           = CUDAMemory::malloc<float>(buffer_size);

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

static struct BufferSizes {
	int trace     [NUM_BOUNCES] = { BATCH_SIZE }; // On the first bounce the TraceBuffer contains exactly BATCH_SIZE Rays
	int diffuse   [NUM_BOUNCES] = { 0 };
	int dielectric[NUM_BOUNCES] = { 0 };
	int glossy    [NUM_BOUNCES] = { 0 };
	int shadow    [NUM_BOUNCES] = { 0 };

	int rays_retired       [NUM_BOUNCES] = { 0 };
	int rays_retired_shadow[NUM_BOUNCES] = { 0 };
} buffer_sizes;

void Pathtracer::init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle) {
	ScopedTimer timer("Pathtracer Initialization");

	CUDAContext::init();

	camera.init(DEG_TO_RAD(110.0f));
	camera.resize(SCREEN_WIDTH, SCREEN_HEIGHT);

	// Init CUDA Module and its Kernel
	module.init("CUDA_Source/Pathtracer.cu", CUDAContext::compute_capability, 64);

	const MeshData * mesh = MeshData::load(scene_name);

	// Set global Material table
	if (mesh->material_count > 0) {
		module.get_global("materials").set_buffer(mesh->materials, mesh->material_count);
	}

	// Set global Texture table
	int texture_count = Texture::textures.size();
	if (texture_count > 0) {
		CUtexObject * tex_objects = new CUtexObject[texture_count];

		for (int i = 0; i < texture_count; i++) {
			const Texture & texture = Texture::textures[i];

			// Create Array on GPU
			CUarray array = CUDAMemory::create_array(
				texture.width,
				texture.height,
				texture.channels,
				texture.get_cuda_array_format()
			);

			// Copy Texture data over to GPU
			CUDAMemory::copy_array(array, texture.get_width_in_bytes(), texture.height, texture.data);

			// Describe the Array to read from
			CUDA_RESOURCE_DESC res_desc = { };
			res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
			res_desc.res.array.hArray = array;

			// Describe how to sample the Texture
			CUDA_TEXTURE_DESC tex_desc = { };
			tex_desc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
			tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB;

			CUDA_RESOURCE_VIEW_DESC view_desc = { };
			view_desc.format = texture.get_cuda_resource_view_format();
			view_desc.width  = texture.get_cuda_resource_view_width();
			view_desc.height = texture.get_cuda_resource_view_height();

			CUDACALL(cuTexObjectCreate(tex_objects + i, &res_desc, &tex_desc, &view_desc));
		}

		module.get_global("textures").set_buffer(tex_objects, texture_count);

		delete [] tex_objects;
	}
	
	// Construct BVH for the Triangle soup
	BVH bvh;
#if BVH_TYPE == BVH_BVH
	bvh = BVHBuilders::bvh(scene_name, mesh);
#else // All other BVH's rely on SBVH
	bvh = BVHBuilders::sbvh(scene_name, mesh);
#endif

	int index_count;
	int primitive_count = bvh.triangle_count;

	int      * indices;
	Triangle * primitives;

#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	module.get_global("bvh_nodes").set_buffer(bvh.nodes, bvh.node_count);

	index_count = bvh.index_count;

	indices    = bvh.indices;
	primitives = bvh.triangles;
#elif BVH_TYPE == BVH_QBVH
	QBVH qbvh = BVHBuilders::qbvh_from_binary_bvh(bvh);
	
	// Set global QBVHNode buffer
	module.get_global("qbvh_nodes").set_buffer(qbvh.nodes, qbvh.node_count);
	
	index_count = bvh.index_count;

	indices    = qbvh.indices;
	primitives = qbvh.triangles;
#elif BVH_TYPE == BVH_CWBVH
	CWBVH cwbvh = BVHBuilders::cwbvh_from_binary_bvh(bvh);

	module.get_global("cwbvh_nodes").set_buffer(cwbvh.nodes, cwbvh.node_count);
	
	index_count = cwbvh.index_count;

	indices    = cwbvh.indices;
	primitives = cwbvh.triangles;
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
	
	CUDATriangle * triangles             = new CUDATriangle[index_count];
	int          * triangle_material_ids = new int         [index_count];

	int * reverse_indices = new int[index_count];

	for (int i = 0; i < index_count; i++) {
		int index = indices[i];

		triangles[i].position_0      = primitives[index].position_0;
		triangles[i].position_edge_1 = primitives[index].position_1 - primitives[index].position_0;
		triangles[i].position_edge_2 = primitives[index].position_2 - primitives[index].position_0;

		triangles[i].normal_0      = primitives[index].normal_0;
		triangles[i].normal_edge_1 = primitives[index].normal_1 - primitives[index].normal_0;
		triangles[i].normal_edge_2 = primitives[index].normal_2 - primitives[index].normal_0;

		triangles[i].tex_coord_0      = primitives[index].tex_coord_0;
		triangles[i].tex_coord_edge_1 = primitives[index].tex_coord_1 - primitives[index].tex_coord_0;
		triangles[i].tex_coord_edge_2 = primitives[index].tex_coord_2 - primitives[index].tex_coord_0;

		triangle_material_ids[i] = primitives[index].material_id;

		reverse_indices[index] = i;
	}

	module.get_global("triangles")            .set_buffer(triangles,             index_count);
	module.get_global("triangle_material_ids").set_buffer(triangle_material_ids, index_count);

	// Create Vertex Buffer for OpenGL containing all Triangles
	vertex_count = primitive_count * 3;
	Vertex * vertices = new Vertex[vertex_count];

	for (int i = 0; i < primitive_count; i++) {
		int index_0 = 3 * i;
		int index_1 = 3 * i + 1;
		int index_2 = 3 * i + 2;

		vertices[index_0].position = primitives[i].position_0;
		vertices[index_1].position = primitives[i].position_1;
		vertices[index_2].position = primitives[i].position_2;

		vertices[index_0].normal = primitives[i].normal_0;
		vertices[index_1].normal = primitives[i].normal_1;
		vertices[index_2].normal = primitives[i].normal_2;

		// Barycentric coordinates
		vertices[index_0].uv = Vector2(0.0f, 0.0f);
		vertices[index_1].uv = Vector2(1.0f, 0.0f);
		vertices[index_2].uv = Vector2(0.0f, 1.0f);

		vertices[index_0].triangle_id = reverse_indices[i];
		vertices[index_1].triangle_id = reverse_indices[i];
		vertices[index_2].triangle_id = reverse_indices[i];
	}

	GLuint vbo;
	glGenBuffers(1, &vbo);

	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vertex_count * sizeof(Vertex), vertices, GL_STATIC_DRAW);

	delete [] vertices;
	
	// Initialize OpenGL Shaders
	shader = Shader::load(DATA_PATH("Shaders/primary_vertex.glsl"), DATA_PATH("Shaders/primary_fragment.glsl"));

	shader.bind();
	uniform_view_projection      = shader.get_uniform("view_projection");
	uniform_view_projection_prev = shader.get_uniform("view_projection_prev");

	// Initialize GBuffers
	gbuffer.init(SCREEN_WIDTH, SCREEN_HEIGHT);

	module.set_texture("gbuffer_normal_and_depth",     CUDAContext::map_gl_texture(gbuffer.buffer_normal_and_depth, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        4);
	module.set_texture("gbuffer_uv",                   CUDAContext::map_gl_texture(gbuffer.buffer_uv,               CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        2);
	module.set_texture("gbuffer_uv_gradient",          CUDAContext::map_gl_texture(gbuffer.buffer_uv_gradient,      CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        4);
	module.set_texture("gbuffer_triangle_id",          CUDAContext::map_gl_texture(gbuffer.buffer_triangle_id,      CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_SIGNED_INT32, 1);
	module.set_texture("gbuffer_screen_position_prev", CUDAContext::map_gl_texture(gbuffer.buffer_motion,           CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        2);
	module.set_texture("gbuffer_depth_gradient",       CUDAContext::map_gl_texture(gbuffer.buffer_z_gradient,       CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        2);

	// Initialize Lights

	struct LightDescription {
		int   index;
		float area;
	};
	std::vector<LightDescription> lights;

	// For every Triangle, check whether it is a Light based on its Material
	for (int i = 0; i < mesh->triangle_count; i++) {
		const Triangle & triangle = mesh->triangles[i];

		if (mesh->materials[triangle.material_id].type == Material::Type::LIGHT) {
			float area = 0.5f * Vector3::length(Vector3::cross(
				mesh->triangles[i].position_1 - mesh->triangles[i].position_0,
				mesh->triangles[i].position_2 - mesh->triangles[i].position_0
			));

			lights.push_back({ reverse_indices[i], area });
		}
	}
	
	int light_count = lights.size();

	if (light_count > 0) {
		// Sort Lights on area
		std::sort(lights.begin(), lights.end(), [](const LightDescription & a, const LightDescription & b) { return a.area < b.area; });

		// Build cumulative table of each Light's area
		int   * light_indices          = new int  [light_count];
		float * light_areas_cumulative = new float[light_count + 1];

		float light_area_total = 0.0f;

		for (int i = 0; i < light_count; i++) {
			light_indices         [i] = lights[i].index;
			light_areas_cumulative[i] = light_area_total;

			light_area_total += lights[i].area;
		} 

		light_areas_cumulative[light_count] = light_area_total;

		module.get_global("light_indices")         .set_buffer(light_indices,          light_count);
		module.get_global("light_areas_cumulative").set_buffer(light_areas_cumulative, light_count + 1);

		module.get_global("light_area_total").set_value(light_area_total);

		delete [] light_indices;
		delete [] light_areas_cumulative;
	}
	
	delete [] triangles;
	delete [] reverse_indices;

	module.get_global("light_count").set_value(light_count);

	// Initialize Sky
	Sky sky;
	sky.init(sky_name);

	module.get_global("sky_size").set_value (sky.size);
	module.get_global("sky_data").set_buffer(sky.data, sky.size * sky.size);
	
	// Set Blue Noise Sampler globals
	module.get_global("sobol_256spp_256d").set_buffer(sobol_256spp_256d);
	module.get_global("scrambling_tile").set_buffer(scrambling_tile);
	module.get_global("ranking_tile").set_buffer(ranking_tile);
	
	// Create Frame Buffers
	module.get_global("frame_buffer_albedo").set_value(CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT).ptr);
	module.get_global("frame_buffer_moment").set_value(CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT).ptr);
	
	ptr_direct       = CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT);
	ptr_indirect     = CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT);
	ptr_direct_alt   = CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT);
	ptr_indirect_alt = CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT);

	module.get_global("frame_buffer_direct")  .set_value(ptr_direct.ptr);
	module.get_global("frame_buffer_indirect").set_value(ptr_indirect.ptr);

	// Set Accumulator to a CUDA resource mapping of the GL frame buffer texture
	module.set_surface("accumulator", CUDAContext::map_gl_texture(frame_buffer_handle, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));

	// Create History Buffers for SVGF
	module.get_global("history_length")          .set_value(CUDAMemory::malloc<int>   (SCREEN_PITCH * SCREEN_HEIGHT).ptr);
	module.get_global("history_direct")          .set_value(CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT).ptr);
	module.get_global("history_indirect")        .set_value(CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT).ptr);
	module.get_global("history_moment")          .set_value(CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT).ptr);
	module.get_global("history_normal_and_depth").set_value(CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT).ptr);
	
	// Create Frame Buffers for Temporal Anti-Aliasing
	module.get_global("taa_frame_prev").set_value(CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT));
	module.get_global("taa_frame_curr").set_value(CUDAMemory::malloc<float4>(SCREEN_PITCH * SCREEN_HEIGHT));

	// Initialize buffers used by Wavefront kernels
	TraceBuffer     ray_buffer_trace;
	MaterialBuffer  ray_buffer_shade_diffuse;
	MaterialBuffer  ray_buffer_shade_dielectric;
	MaterialBuffer  ray_buffer_shade_glossy;
	ShadowRayBuffer ray_buffer_shadow;

	ray_buffer_trace           .init(BATCH_SIZE);
	ray_buffer_shade_diffuse   .init(BATCH_SIZE);
	ray_buffer_shade_dielectric.init(BATCH_SIZE);
	ray_buffer_shade_glossy    .init(BATCH_SIZE);
	ray_buffer_shadow          .init(BATCH_SIZE);

	module.get_global("ray_buffer_trace")           .set_value(ray_buffer_trace);
	module.get_global("ray_buffer_shade_diffuse")   .set_value(ray_buffer_shade_diffuse);
	module.get_global("ray_buffer_shade_dielectric").set_value(ray_buffer_shade_dielectric);
	module.get_global("ray_buffer_shade_glossy")    .set_value(ray_buffer_shade_glossy);
	module.get_global("ray_buffer_shadow")          .set_value(ray_buffer_shadow);

	global_buffer_sizes = module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(buffer_sizes);

	global_svgf_settings = module.get_global("svgf_settings");

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
	kernel_shadow_trace    .init(&module, "kernel_shadow_trace");
	kernel_svgf_temporal   .init(&module, "kernel_svgf_temporal");
	kernel_svgf_variance   .init(&module, "kernel_svgf_variance");
	kernel_svgf_atrous     .init(&module, "kernel_svgf_atrous");
	kernel_svgf_finalize   .init(&module, "kernel_svgf_finalize");
	kernel_taa             .init(&module, "kernel_taa");
	kernel_taa_finalize    .init(&module, "kernel_taa_finalize");
	kernel_accumulate      .init(&module, "kernel_accumulate");

	// Set Block dimensions for all Kernels
	kernel_svgf_temporal.occupancy_max_block_size_2d();
	kernel_svgf_variance.occupancy_max_block_size_2d();
	kernel_svgf_atrous  .occupancy_max_block_size_2d();
	kernel_svgf_finalize.occupancy_max_block_size_2d();
	kernel_taa          .occupancy_max_block_size_2d();
	kernel_taa_finalize .occupancy_max_block_size_2d();
	kernel_accumulate   .occupancy_max_block_size_2d();

	kernel_primary         .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_generate        .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_sort            .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_diffuse   .set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_dielectric.set_block_dim(WARP_SIZE * 2, 1, 1);
	kernel_shade_glossy    .set_block_dim(WARP_SIZE * 2, 1, 1);
	
	kernel_trace       .set_block_dim(WARP_SIZE,        TRACE_BLOCK_Y, 1);
	kernel_shadow_trace.set_block_dim(WARP_SIZE, SHADOW_TRACE_BLOCK_Y, 1);

	// Set Grid dimensions for all Kernels
	kernel_svgf_temporal.set_grid_dim(SCREEN_PITCH / kernel_svgf_temporal.block_dim_x, (SCREEN_HEIGHT + kernel_svgf_temporal.block_dim_y - 1) / kernel_svgf_temporal.block_dim_y, 1);
	kernel_svgf_variance.set_grid_dim(SCREEN_PITCH / kernel_svgf_variance.block_dim_x, (SCREEN_HEIGHT + kernel_svgf_variance.block_dim_y - 1) / kernel_svgf_variance.block_dim_y, 1);
	kernel_svgf_atrous  .set_grid_dim(SCREEN_PITCH / kernel_svgf_atrous  .block_dim_x, (SCREEN_HEIGHT + kernel_svgf_atrous  .block_dim_y - 1) / kernel_svgf_atrous  .block_dim_y, 1);
	kernel_svgf_finalize.set_grid_dim(SCREEN_PITCH / kernel_svgf_finalize.block_dim_x, (SCREEN_HEIGHT + kernel_svgf_finalize.block_dim_y - 1) / kernel_svgf_finalize.block_dim_y, 1);
	kernel_taa          .set_grid_dim(SCREEN_PITCH / kernel_taa          .block_dim_x, (SCREEN_HEIGHT + kernel_taa          .block_dim_y - 1) / kernel_taa          .block_dim_y, 1);
	kernel_taa_finalize .set_grid_dim(SCREEN_PITCH / kernel_taa_finalize .block_dim_x, (SCREEN_HEIGHT + kernel_taa_finalize .block_dim_y - 1) / kernel_taa_finalize .block_dim_y, 1);
	kernel_accumulate   .set_grid_dim(SCREEN_PITCH / kernel_accumulate   .block_dim_x, (SCREEN_HEIGHT + kernel_accumulate   .block_dim_y - 1) / kernel_accumulate   .block_dim_y, 1);

	kernel_primary         .set_grid_dim(BATCH_SIZE / kernel_primary         .block_dim_x, 1, 1);
	kernel_generate        .set_grid_dim(BATCH_SIZE / kernel_generate        .block_dim_x, 1, 1);
	kernel_sort            .set_grid_dim(BATCH_SIZE / kernel_sort            .block_dim_x, 1, 1);
	kernel_shade_diffuse   .set_grid_dim(BATCH_SIZE / kernel_shade_diffuse   .block_dim_x, 1, 1);
	kernel_shade_dielectric.set_grid_dim(BATCH_SIZE / kernel_shade_dielectric.block_dim_x, 1, 1);
	kernel_shade_glossy    .set_grid_dim(BATCH_SIZE / kernel_shade_glossy    .block_dim_x, 1, 1);
	
	kernel_trace       .set_grid_dim(32, 32, 1);
	kernel_shadow_trace.set_grid_dim(32, 32, 1);

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

	event_taa       .init("Post", "TAA");
	event_accumulate.init("Post", "Accumulate");

	event_end.init("END", "END");

	scene_has_diffuse    = false;
	scene_has_dielectric = false;
	scene_has_glossy     = false;
	scene_has_lights     = false;

	// Check properties of the Scene, so we know which kernels are required
	for (int i = 0; i < mesh->material_count; i++) {
		switch (mesh->materials[i].type) {
			case Material::Type::DIFFUSE:    scene_has_diffuse    = true; break;
			case Material::Type::DIELECTRIC: scene_has_dielectric = true; break;
			case Material::Type::GLOSSY:     scene_has_glossy     = true; break;
			case Material::Type::LIGHT:      scene_has_lights     = true; break;
		}
	}

	printf("\nScene info:\ndiffuse:    %s\ndielectric: %s\nglossy:     %s\nlights:     %s\n\n", 
		scene_has_diffuse    ? "yes" : "no",
		scene_has_dielectric ? "yes" : "no",
		scene_has_glossy     ? "yes" : "no",
		scene_has_lights     ? "yes" : "no"
	);

	int    scene_name_length = strlen(scene_name);
	char * scene_name_lower  = reinterpret_cast<char *>(_malloca(scene_name_length + 1));

	for (int i = 0; i < scene_name_length + 1; i++) {
		scene_name_lower[i] = tolower(scene_name[i]);
	}

	// Initialize Camera position/orientation based on the Scene name
	if (strstr(scene_name_lower, "pica.obj")) {
		camera.position = Vector3(-7.640668f, 16.404673f, 17.845022f);
		camera.rotation = Quaternion(-0.256006f, -0.069205f, -0.018378f, 0.964019f);	
	} else if (strstr(scene_name_lower, "sponza.obj")) {
		camera.position = Vector3(116.927467f, 15.586369f, -2.997146f);
		camera.rotation = Quaternion(0.000000f, 0.692966f, 0.000000f, 0.720969f);
	} else if (strstr(scene_name_lower, "scene.obj")) {
		camera.position = Vector3(-0.126737f, 0.613379f, 3.716630f);
		camera.rotation = Quaternion(-0.107255f, -0.002421f, 0.000262f, -0.994227f);
	} else if (strstr(scene_name_lower, "cornellbox.obj")) {
		camera.position = Vector3(0.528027f, 1.004323f, -0.774033f);
		camera.rotation = Quaternion(0.035059f, -0.963870f, 0.208413f, 0.162142f);
	} else if (strstr(scene_name_lower, "glossy.obj")) {
		camera.position = Vector3(-5.438800f, 5.910520f, -7.185338f);
		camera.rotation = Quaternion(0.242396f, 0.716713f, 0.298666f, -0.581683f);
	} else if (strstr(scene_name_lower, "bunny.obj")) {
		camera.position = Vector3(-27.662603f, 26.719784f, -15.835464f);
		camera.rotation = Quaternion(0.076750f, 0.900785f, 0.177892f, -0.388638f);
	} else if (strstr(scene_name_lower, "test.obj")) {
		camera.position = Vector3(4.157419f, 4.996608f, 8.337481f);
		camera.rotation = Quaternion(0.000000f, 0.310172f, 0.000000f, 0.950679f);
	} else if (strstr(scene_name_lower, "bistro.obj")) {
		camera.position = Vector3(-13.665823f, 2.480730f, -2.920546f);
		camera.rotation = Quaternion(0.000000f, -0.772662f, 0.000000f, 0.634818f);
	} else if (strstr(scene_name_lower, "rungholt.obj")) {
		camera.position = Vector3(-22.413084f, 18.681219f, -23.566196f);
		camera.rotation = Quaternion(0.000000f, 0.716948f, 0.000000f, -0.697125f);
	} else {
		camera.position = Vector3(1.272743f, 3.097532f, -3.189943f);
		camera.rotation = Quaternion(0.000000f, 0.995683f, 0.000000f, -0.092814f);
	}

	_freea(scene_name_lower);
}

void Pathtracer::update(float delta, const unsigned char * keys) {
	bool jitter = enable_taa;
	camera.update(delta, keys, jitter);

	if (settings_changed) {
		frames_since_camera_moved = 0;

		global_svgf_settings.set_value(svgf_settings);
	} else if (enable_svgf) {
		frames_since_camera_moved = (frames_since_camera_moved + 1) & 255;
	} else if (camera.moved) {
		frames_since_camera_moved = 0;
	} else {
		frames_since_camera_moved++;
	}
}

#define RECORD_EVENT(e) e.record(); events.push_back(&e);

void Pathtracer::render() {
	events.clear();

	if (enable_rasterization) {
		gbuffer.bind();

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		shader.bind();

		glUniformMatrix4fv(uniform_view_projection,      1, GL_TRUE, reinterpret_cast<const GLfloat *>(&camera.view_projection));
		glUniformMatrix4fv(uniform_view_projection_prev, 1, GL_TRUE, reinterpret_cast<const GLfloat *>(&camera.view_projection_prev));

		glEnableVertexAttribArray(0);
		glEnableVertexAttribArray(1);
		glEnableVertexAttribArray(2);
		glEnableVertexAttribArray(3);

		glVertexAttribPointer (0, 3, GL_FLOAT, false, sizeof(Vertex), reinterpret_cast<const GLvoid *>(offsetof(Vertex, position)));
		glVertexAttribPointer (1, 3, GL_FLOAT, false, sizeof(Vertex), reinterpret_cast<const GLvoid *>(offsetof(Vertex, normal)));
		glVertexAttribPointer (2, 2, GL_FLOAT, false, sizeof(Vertex), reinterpret_cast<const GLvoid *>(offsetof(Vertex, uv)));
		glVertexAttribIPointer(3, 1, GL_INT,          sizeof(Vertex), reinterpret_cast<const GLvoid *>(offsetof(Vertex, triangle_id)));

		glDrawArrays(GL_TRIANGLES, 0, vertex_count);

		glDisableVertexAttribArray(3);
		glDisableVertexAttribArray(2);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(0);

		shader .unbind();
		gbuffer.unbind();

		glFinish();
	}

	int pixels_left = PIXEL_COUNT;

	// Render in batches of BATCH_SIZE pixels at a time
	while (pixels_left > 0) {
		int pixel_offset = PIXEL_COUNT - pixels_left;
		int pixel_count  = pixels_left > BATCH_SIZE ? BATCH_SIZE : pixels_left;

		RECORD_EVENT(event_primary);

		if (enable_rasterization) {
			bool jitter = enable_taa;

			// Convert rasterized GBuffers into primary Rays
			kernel_primary.execute(
				rand(),
				frames_since_camera_moved,
				pixel_offset,
				pixel_count,
				jitter,
				camera.position,
				camera.bottom_left_corner_rotated,
				camera.x_axis_rotated,
				camera.y_axis_rotated
			);
		} else {
			// Generate primary Rays from the current Camera orientation
			kernel_generate.execute(
				rand(),
				frames_since_camera_moved,
				pixel_offset,
				pixel_count,
				camera.position, 
				camera.bottom_left_corner_rotated, 
				camera.x_axis_rotated, 
				camera.y_axis_rotated
			);
		}

		for (int bounce = 0; bounce < NUM_BOUNCES; bounce++) {
			// When rasterizing primary rays we can skip tracing rays on bounce 0
			if (!(bounce == 0 && enable_rasterization)) {
				// Extend all Rays that are still alive to their next Triangle intersection
				RECORD_EVENT(event_trace[bounce]);
				kernel_trace.execute(bounce);

				RECORD_EVENT(event_sort[bounce]);
				kernel_sort.execute(rand(), bounce);
			}

			// Process the various Material types in different Kernels
			if (scene_has_diffuse) {
				RECORD_EVENT(event_shade_diffuse[bounce]);
				kernel_shade_diffuse.execute(rand(), bounce, frames_since_camera_moved);
			}

			if (scene_has_dielectric) {
				RECORD_EVENT(event_shade_dielectric[bounce]);
				kernel_shade_dielectric.execute(rand(), bounce);
			}

			if (scene_has_glossy) {
				RECORD_EVENT(event_shade_glossy[bounce]);
				kernel_shade_glossy.execute(rand(), bounce, frames_since_camera_moved);
			}

			// Trace shadow Rays
			if (scene_has_lights) {
				RECORD_EVENT(event_shadow_trace[bounce]);
				kernel_shadow_trace.execute(bounce);
			}
		}

		pixels_left -= BATCH_SIZE;

		if (pixels_left > 0) {
			// Set buffer sizes to appropriate pixel count for next Batch
			buffer_sizes.trace[0] = pixels_left > BATCH_SIZE ? BATCH_SIZE : pixels_left;
			global_buffer_sizes.set_value(buffer_sizes);
		}
	}

	if (enable_svgf) {
		// Integrate temporally
		RECORD_EVENT(event_svgf_temporal);
		kernel_svgf_temporal.execute();
	
		CUdeviceptr direct_in    = ptr_direct.ptr;
		CUdeviceptr indirect_in  = ptr_indirect.ptr;
		CUdeviceptr direct_out   = ptr_direct_alt.ptr;
		CUdeviceptr indirect_out = ptr_indirect_alt.ptr;

		if (enable_spatial_variance) {
			// Estimate Variance spatially
			RECORD_EVENT(event_svgf_variance);
			kernel_svgf_variance.execute(direct_in, indirect_in, direct_out, indirect_out);
		} else {
			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);
		}

		// À-Trous Filter
		for (int i = 0; i < svgf_settings.atrous_iterations; i++) {
			int step_size = 1 << i;
				
			// Ping-Pong the Frame Buffers
			std::swap(direct_in,   direct_out);
			std::swap(indirect_in, indirect_out);

			RECORD_EVENT(event_svgf_atrous[i]);
			kernel_svgf_atrous.execute(direct_in, indirect_in, direct_out, indirect_out, step_size);
		}

		RECORD_EVENT(event_svgf_finalize);
		kernel_svgf_finalize.execute(enable_albedo, direct_out, indirect_out);

		if (enable_taa) {
			RECORD_EVENT(event_taa);

			kernel_taa         .execute();
			kernel_taa_finalize.execute();
		}
	} else {
		RECORD_EVENT(event_accumulate);
		kernel_accumulate.execute(!enable_albedo, float(frames_since_camera_moved));
	}
	
	RECORD_EVENT(event_end);
	
	// Reset buffer sizes to default for next frame
	buffer_sizes.trace[0] = BATCH_SIZE;
	global_buffer_sizes.set_value(buffer_sizes);
}
