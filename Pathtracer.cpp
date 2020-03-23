#include "Pathtracer.h"

#include <filesystem>

#include "CUDAContext.h"

#include "MeshData.h"
#include "BVH.h"
#include "MBVH.h"

#include "Sky.h"

#include "BlueNoise.h"

#include "ScopedTimer.h"

struct Vertex {
	Vector3 position;
	Vector3 normal;
	Vector2 uv;
	int     triangle_id;
};

static struct ExtendBuffer {
	CUDAMemory::Ptr<float> origin_x;
	CUDAMemory::Ptr<float> origin_y;
	CUDAMemory::Ptr<float> origin_z;
	CUDAMemory::Ptr<float> direction_x;
	CUDAMemory::Ptr<float> direction_y;
	CUDAMemory::Ptr<float> direction_z;
	
	CUDAMemory::Ptr<int>   pixel_index;
	CUDAMemory::Ptr<float> throughput_x;
	CUDAMemory::Ptr<float> throughput_y;
	CUDAMemory::Ptr<float> throughput_z;

	CUDAMemory::Ptr<char>  last_material_type;
	CUDAMemory::Ptr<float> last_pdf;

	inline void init(int buffer_size) {
		origin_x    = CUDAMemory::malloc<float>(buffer_size);
		origin_y    = CUDAMemory::malloc<float>(buffer_size);
		origin_z    = CUDAMemory::malloc<float>(buffer_size);
		direction_x = CUDAMemory::malloc<float>(buffer_size);
		direction_y = CUDAMemory::malloc<float>(buffer_size);
		direction_z = CUDAMemory::malloc<float>(buffer_size);

		pixel_index   = CUDAMemory::malloc<int>(buffer_size);
		throughput_x  = CUDAMemory::malloc<float>(buffer_size);
		throughput_y  = CUDAMemory::malloc<float>(buffer_size);
		throughput_z  = CUDAMemory::malloc<float>(buffer_size);

		last_material_type = CUDAMemory::malloc<char>(buffer_size);
		last_pdf           = CUDAMemory::malloc<float>(buffer_size);
	}
};
	
static struct MaterialBuffer {
	CUDAMemory::Ptr<float> direction_x;
	CUDAMemory::Ptr<float> direction_y;
	CUDAMemory::Ptr<float> direction_z;
	
	CUDAMemory::Ptr<int> triangle_id;
	CUDAMemory::Ptr<float> u;
	CUDAMemory::Ptr<float> v;

	CUDAMemory::Ptr<int>   pixel_index;
	CUDAMemory::Ptr<float> throughput_x;
	CUDAMemory::Ptr<float> throughput_y;
	CUDAMemory::Ptr<float> throughput_z;

	inline void init(int buffer_size) {
		direction_x = CUDAMemory::malloc<float>(buffer_size);
		direction_y = CUDAMemory::malloc<float>(buffer_size);
		direction_z = CUDAMemory::malloc<float>(buffer_size);

		triangle_id = CUDAMemory::malloc<int>(buffer_size);
		u = CUDAMemory::malloc<float>(buffer_size);
		v = CUDAMemory::malloc<float>(buffer_size);

		pixel_index   = CUDAMemory::malloc<int>(buffer_size);
		throughput_x  = CUDAMemory::malloc<float>(buffer_size);
		throughput_y  = CUDAMemory::malloc<float>(buffer_size);
		throughput_z  = CUDAMemory::malloc<float>(buffer_size);
	}
};
	
static struct ShadowRayBuffer {
	CUDAMemory::Ptr<float> direction_x;
	CUDAMemory::Ptr<float> direction_y;
	CUDAMemory::Ptr<float> direction_z;

	CUDAMemory::Ptr<int> triangle_id;
	CUDAMemory::Ptr<float> u;
	CUDAMemory::Ptr<float> v;

	CUDAMemory::Ptr<int> pixel_index;
	CUDAMemory::Ptr<float> throughput_x;
	CUDAMemory::Ptr<float> throughput_y;
	CUDAMemory::Ptr<float> throughput_z;

	inline void init(int buffer_size) {
		direction_x = CUDAMemory::malloc<float>(buffer_size);
		direction_y = CUDAMemory::malloc<float>(buffer_size);
		direction_z = CUDAMemory::malloc<float>(buffer_size);

		triangle_id = CUDAMemory::malloc<int>(buffer_size);
		u = CUDAMemory::malloc<float>(buffer_size);
		v = CUDAMemory::malloc<float>(buffer_size);

		pixel_index  = CUDAMemory::malloc<int>(buffer_size);
		throughput_x = CUDAMemory::malloc<float>(buffer_size);
		throughput_y = CUDAMemory::malloc<float>(buffer_size);
		throughput_z = CUDAMemory::malloc<float>(buffer_size);
	}
};

static struct BufferSizes {
	int N_extend    [NUM_BOUNCES] = { PIXEL_COUNT }; // On the first bounce the ExtendBuffer contains exactly PIXEL_COUNT Rays
	int N_diffuse   [NUM_BOUNCES] = { 0 };
	int N_dielectric[NUM_BOUNCES] = { 0 };
	int N_glossy    [NUM_BOUNCES] = { 0 };
	int N_shadow    [NUM_BOUNCES] = { 0 };
};

static BufferSizes buffer_sizes;

void Pathtracer::init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle) {
	CUDAContext::init();

	camera.init(DEG_TO_RAD(110.0f));
	camera.resize(SCREEN_WIDTH, SCREEN_HEIGHT);

	// Init CUDA Module and its Kernel
	module.init("CUDA_Source/Pathtracer.cu", CUDAContext::compute_capability);

	const MeshData * mesh = MeshData::load(scene_name);

	if (mesh->material_count > MAX_MATERIALS || Texture::texture_count > MAX_TEXTURES) abort();

	// Set global Material table
	module.get_global("materials").set_buffer(mesh->materials, mesh->material_count);

	// Set global Texture table
	if (Texture::texture_count > 0) {
		CUtexObject * tex_objects = new CUtexObject[Texture::texture_count];

		for (int i = 0; i < Texture::texture_count; i++) {
			CUarray array = CUDAMemory::create_array(Texture::textures[i].width, Texture::textures[i].height, Texture::textures[i].channels, CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8);
		
			CUDAMemory::copy_array(array, Texture::textures[i].channels * Texture::textures[i].width, Texture::textures[i].height, Texture::textures[i].data);

			// Describe the Array to read from
			CUDA_RESOURCE_DESC res_desc = { };
			res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
			res_desc.res.array.hArray = array;
		
			// Describe how to sample the Texture
			CUDA_TEXTURE_DESC tex_desc = { };
			tex_desc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_POINT;
			tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES | CU_TRSF_SRGB;
			
			CUDACALL(cuTexObjectCreate(tex_objects + i, &res_desc, &tex_desc, nullptr));
		}

		module.get_global("textures").set_buffer(tex_objects, Texture::texture_count);

		delete [] tex_objects;
	}

	// Construct BVH for the Triangle soup
	BVH<Triangle> bvh;

	std::string bvh_filename = std::string(scene_name) + ".bvh";
	if (std::filesystem::exists(bvh_filename)) {
		printf("Loading BVH %s from disk.\n", bvh_filename.c_str());

		bvh.load_from_disk(bvh_filename.c_str());
	} else {
		bvh.init(mesh->triangle_count);

		memcpy(bvh.primitives, mesh->triangles, mesh->triangle_count * sizeof(Triangle));

		for (int i = 0; i < bvh.primitive_count; i++) {
			Vector3 vertices[3] = { 
				bvh.primitives[i].position_0, 
				bvh.primitives[i].position_1, 
				bvh.primitives[i].position_2
			};
			bvh.primitives[i].aabb = AABB::from_points(vertices, 3);
		}

		{
			ScopedTimer timer("SBVH Construction");

			bvh.build_sbvh();
		}

		bvh.save_to_disk(bvh_filename.c_str());
	}

	MBVH<Triangle> mbvh;
	mbvh.init(bvh);

	// Flatten the Primitives array so that we don't need the indices array as an indirection to index it
	// (This does mean more memory consumption)
	Triangle * flat_triangles  = new Triangle[mbvh.leaf_count];
	int      * reverse_indices = new int     [mbvh.leaf_count];

	for (int i = 0; i < mbvh.leaf_count; i++) {
		flat_triangles[i] = mbvh.primitives[mbvh.indices[i]];

		reverse_indices[mbvh.indices[i]] = i;
	}

	// Allocate Triangles in SoA format
	Vector3 * triangles_position0      = new Vector3[mbvh.leaf_count];
	Vector3 * triangles_position_edge1 = new Vector3[mbvh.leaf_count];
	Vector3 * triangles_position_edge2 = new Vector3[mbvh.leaf_count];

	Vector3 * triangles_normal0      = new Vector3[mbvh.leaf_count];
	Vector3 * triangles_normal_edge1 = new Vector3[mbvh.leaf_count];
	Vector3 * triangles_normal_edge2 = new Vector3[mbvh.leaf_count]; 
	
	Vector2 * triangles_tex_coord0      = new Vector2[mbvh.leaf_count];
	Vector2 * triangles_tex_coord_edge1 = new Vector2[mbvh.leaf_count];
	Vector2 * triangles_tex_coord_edge2 = new Vector2[mbvh.leaf_count];

	int * triangles_material_id = new int[mbvh.leaf_count];

	for (int i = 0; i < mbvh.leaf_count; i++) {
		triangles_position0[i]      = flat_triangles[i].position_0;
		triangles_position_edge1[i] = flat_triangles[i].position_1 - flat_triangles[i].position_0;
		triangles_position_edge2[i] = flat_triangles[i].position_2 - flat_triangles[i].position_0;

		triangles_normal0[i]      = flat_triangles[i].normal_0;
		triangles_normal_edge1[i] = flat_triangles[i].normal_1 - flat_triangles[i].normal_0;
		triangles_normal_edge2[i] = flat_triangles[i].normal_2 - flat_triangles[i].normal_0;

		triangles_tex_coord0[i]      = flat_triangles[i].tex_coord_0;
		triangles_tex_coord_edge1[i] = flat_triangles[i].tex_coord_1 - flat_triangles[i].tex_coord_0;
		triangles_tex_coord_edge2[i] = flat_triangles[i].tex_coord_2 - flat_triangles[i].tex_coord_0;

		triangles_material_id[i] = flat_triangles[i].material_id;
	}

	delete [] flat_triangles;

	// Set global Triangle buffers
	module.get_global("triangles_position0"     ).set_buffer(triangles_position0,      mbvh.leaf_count);
	module.get_global("triangles_position_edge1").set_buffer(triangles_position_edge1, mbvh.leaf_count);
	module.get_global("triangles_position_edge2").set_buffer(triangles_position_edge2, mbvh.leaf_count);

	module.get_global("triangles_normal0"     ).set_buffer(triangles_normal0,      mbvh.leaf_count);
	module.get_global("triangles_normal_edge1").set_buffer(triangles_normal_edge1, mbvh.leaf_count);
	module.get_global("triangles_normal_edge2").set_buffer(triangles_normal_edge2, mbvh.leaf_count);

	module.get_global("triangles_tex_coord0"     ).set_buffer(triangles_tex_coord0,      mbvh.leaf_count);
	module.get_global("triangles_tex_coord_edge1").set_buffer(triangles_tex_coord_edge1, mbvh.leaf_count);
	module.get_global("triangles_tex_coord_edge2").set_buffer(triangles_tex_coord_edge2, mbvh.leaf_count);

	module.get_global("triangles_material_id").set_buffer(triangles_material_id, mbvh.leaf_count);

	// Clean up buffers on Host side
	delete [] triangles_position0;  
	delete [] triangles_position_edge1;
	delete [] triangles_position_edge2;
	delete [] triangles_normal0;
	delete [] triangles_normal_edge1;
	delete [] triangles_normal_edge2;
	delete [] triangles_tex_coord0;  
	delete [] triangles_tex_coord_edge1;
	delete [] triangles_tex_coord_edge2;
	delete [] triangles_material_id;

	vertex_count = mbvh.primitive_count * 3;
	Vertex * vertices = new Vertex[vertex_count];

	for (int i = 0; i < mbvh.primitive_count; i++) {
		int index_0 = 3 * i;
		int index_1 = 3 * i + 1;
		int index_2 = 3 * i + 2;

		vertices[index_0].position = mbvh.primitives[i].position_0;
		vertices[index_1].position = mbvh.primitives[i].position_1;
		vertices[index_2].position = mbvh.primitives[i].position_2;

		vertices[index_0].normal = mbvh.primitives[i].normal_0;
		vertices[index_1].normal = mbvh.primitives[i].normal_1;
		vertices[index_2].normal = mbvh.primitives[i].normal_2;

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
	
	shader = Shader::load(DATA_PATH("Shaders/primary_vertex.glsl"), DATA_PATH("Shaders/primary_fragment.glsl"));

	shader.bind();
	uniform_view_projection      = shader.get_uniform("view_projection");
	uniform_view_projection_prev = shader.get_uniform("view_projection_prev");

	gbuffer.init(SCREEN_WIDTH, SCREEN_HEIGHT);

	module.set_texture("gbuffer_position",       CUDAContext::map_gl_texture(gbuffer.buffer_position,    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        4);
	module.set_texture("gbuffer_normal",         CUDAContext::map_gl_texture(gbuffer.buffer_normal,      CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        4);
	module.set_texture("gbuffer_uv",             CUDAContext::map_gl_texture(gbuffer.buffer_uv,          CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        2);
	module.set_texture("gbuffer_triangle_id",    CUDAContext::map_gl_texture(gbuffer.buffer_triangle_id, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_SIGNED_INT32, 1);
	module.set_texture("gbuffer_motion",         CUDAContext::map_gl_texture(gbuffer.buffer_motion,      CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        2);
	module.set_texture("gbuffer_depth",          CUDAContext::map_gl_texture(gbuffer.buffer_z,           CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        1);
	module.set_texture("gbuffer_depth_gradient", CUDAContext::map_gl_texture(gbuffer.buffer_z_gradient,  CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY), CU_TR_FILTER_MODE_POINT, CU_AD_FORMAT_FLOAT,        2);

	int * light_indices = new int[mesh->triangle_count];
	int   light_count = 0;

	// For every Triangle, check whether it is a Light based on its Material
	for (int i = 0; i < mesh->triangle_count; i++) {
		const Triangle & triangle = mesh->triangles[i];

		if (mesh->materials[triangle.material_id].type == Material::Type::LIGHT) {
			light_indices[light_count++] = reverse_indices[i];
		}
	}
	
	if (light_count > 0) {
		module.get_global("light_indices").set_buffer(light_indices, light_count);
	}

	delete [] light_indices;
	delete [] reverse_indices;

	module.get_global("light_count").set_value(light_count);

	// Set global MBVHNode buffer
	module.get_global("mbvh_nodes").set_buffer(mbvh.nodes, mbvh.node_count);

	// Set Sky globals
	Sky sky;
	sky.init(sky_name);

	module.get_global("sky_size").set_value(sky.size);
	module.get_global("sky_data").set_buffer(sky.data, sky.size * sky.size);
	
	// Set Blue Noise Sampler globals
	module.get_global("sobol_256spp_256d").set_buffer(sobol_256spp_256d);
	module.get_global("scrambling_tile").set_buffer(scrambling_tile);
	module.get_global("ranking_tile").set_buffer(ranking_tile);
	
	// Create Frame Buffers
	module.get_global("frame_buffer_albedo").set_value(CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4).ptr);
	module.get_global("frame_buffer_moment").set_value(CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4).ptr);
	
	ptr_direct       = CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4);
	ptr_indirect     = CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4);
	ptr_direct_alt   = CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4);
	ptr_indirect_alt = CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4);

	module.get_global("frame_buffer_direct").set_value(ptr_direct.ptr);
	module.get_global("frame_buffer_indirect").set_value(ptr_indirect.ptr);

	// Set Accumulator to a CUDA resource mapping of the GL frame buffer texture
	module.set_surface("accumulator", CUDAContext::map_gl_texture(frame_buffer_handle, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));

	// Create History Buffers 
	module.get_global("history_length").set_value     (CUDAMemory::malloc<int>  (SCREEN_WIDTH * SCREEN_HEIGHT    ).ptr);
	module.get_global("history_direct").set_value     (CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4).ptr);
	module.get_global("history_indirect").set_value   (CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4).ptr);
	module.get_global("history_moment").set_value     (CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4).ptr);
	module.get_global("history_position").set_value   (CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4).ptr);
	module.get_global("history_normal").set_value     (CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT * 4).ptr);
	module.get_global("history_triangle_id").set_value(CUDAMemory::malloc<int>  (SCREEN_WIDTH * SCREEN_HEIGHT    ).ptr);
	module.get_global("history_depth").set_value      (CUDAMemory::malloc<float>(SCREEN_WIDTH * SCREEN_HEIGHT    ).ptr);

	ExtendBuffer    ray_buffer_extend;
	MaterialBuffer  ray_buffer_shade_diffuse;
	MaterialBuffer  ray_buffer_shade_dielectric;
	MaterialBuffer  ray_buffer_shade_glossy;
	ShadowRayBuffer ray_buffer_connect;

	ray_buffer_extend.init          (PIXEL_COUNT);
	ray_buffer_shade_diffuse.init   (PIXEL_COUNT);
	ray_buffer_shade_dielectric.init(PIXEL_COUNT);
	ray_buffer_shade_glossy.init    (PIXEL_COUNT);
	ray_buffer_connect.init         (PIXEL_COUNT);

	CUDACALL(cuStreamCreate(&memcpy_stream, CU_STREAM_NON_BLOCKING));

	module.get_global("ray_buffer_extend").set_value          (ray_buffer_extend);
	module.get_global("ray_buffer_shade_diffuse").set_value   (ray_buffer_shade_diffuse);
	module.get_global("ray_buffer_shade_dielectric").set_value(ray_buffer_shade_dielectric);
	module.get_global("ray_buffer_shade_glossy").set_value    (ray_buffer_shade_glossy);
	module.get_global("ray_buffer_connect").set_value         (ray_buffer_connect);

	global_buffer_sizes = module.get_global("buffer_sizes");
	global_buffer_sizes.set_value(buffer_sizes);

	kernel_primary.init         (&module, "kernel_primary");
	kernel_generate.init        (&module, "kernel_generate");
	kernel_extend.init          (&module, "kernel_extend");
	kernel_shade_diffuse.init   (&module, "kernel_shade_diffuse");
	kernel_shade_dielectric.init(&module, "kernel_shade_dielectric");
	kernel_shade_glossy.init    (&module, "kernel_shade_glossy");
	kernel_connect.init         (&module, "kernel_connect");
	kernel_temporal.init        (&module, "kernel_temporal");
	kernel_atrous.init          (&module, "kernel_atrous");
	kernel_finalize.init        (&module, "kernel_finalize");

	kernel_primary.set_block_dim         ( 32, 4, 1);
	kernel_generate.set_block_dim        (128, 1, 1);
	kernel_extend.set_block_dim          (128, 1, 1);
	kernel_shade_diffuse.set_block_dim   (128, 1, 1);
	kernel_shade_dielectric.set_block_dim(128, 1, 1);
	kernel_shade_glossy.set_block_dim    (128, 1, 1);
	kernel_connect.set_block_dim         (128, 1, 1);
	kernel_temporal.set_block_dim        ( 32, 4, 1);
	kernel_atrous.set_block_dim          ( 32, 4, 1);
	kernel_finalize.set_block_dim        ( 32, 4, 1);

	kernel_primary.set_grid_dim(
		(SCREEN_WIDTH  + kernel_primary.block_dim_x - 1) / kernel_primary.block_dim_x, 
		(SCREEN_HEIGHT + kernel_primary.block_dim_y - 1) / kernel_primary.block_dim_y,
		1
	);
	kernel_generate.set_grid_dim        (PIXEL_COUNT / kernel_generate.block_dim_x,         1, 1);
	kernel_extend.set_grid_dim          (PIXEL_COUNT / kernel_extend.block_dim_x,           1, 1);
	kernel_shade_diffuse.set_grid_dim   (PIXEL_COUNT / kernel_shade_diffuse.block_dim_x,    1, 1);
	kernel_shade_dielectric.set_grid_dim(PIXEL_COUNT / kernel_shade_dielectric.block_dim_x, 1, 1);
	kernel_shade_glossy.set_grid_dim    (PIXEL_COUNT / kernel_shade_glossy.block_dim_x,     1, 1);
	kernel_connect.set_grid_dim         (PIXEL_COUNT / kernel_connect.block_dim_x,          1, 1);
	kernel_temporal.set_grid_dim(
		(SCREEN_WIDTH  + kernel_temporal.block_dim_x - 1) / kernel_temporal.block_dim_x, 
		(SCREEN_HEIGHT + kernel_temporal.block_dim_y - 1) / kernel_temporal.block_dim_y,
		1
	);
	kernel_atrous.set_grid_dim(
		(SCREEN_WIDTH  + kernel_atrous.block_dim_x - 1) / kernel_atrous.block_dim_x, 
		(SCREEN_HEIGHT + kernel_atrous.block_dim_y - 1) / kernel_atrous.block_dim_y,
		1
	);
	kernel_finalize.set_grid_dim(
		(SCREEN_WIDTH  + kernel_finalize.block_dim_x - 1) / kernel_finalize.block_dim_x, 
		(SCREEN_HEIGHT + kernel_finalize.block_dim_y - 1) / kernel_finalize.block_dim_y,
		1
	);

	if (strcmp(scene_name, DATA_PATH("pica/pica.obj")) == 0) {
		camera.position = Vector3(-14.875896f, 5.407789f, -22.486183f);
		camera.rotation = Quaternion(0.000000f, 0.980876f, 0.000000f, 0.194635f);
	} else if (strcmp(scene_name, DATA_PATH("sponza/sponza.obj")) == 0) {
		camera.position = Vector3(2.698714f, 39.508224f, -15.633610f);
		camera.rotation = Quaternion(0.000000f, 0.891950f, 0.000000f, 0.452135f);
	} else if (strcmp(scene_name, DATA_PATH("scene.obj")) == 0) {
		camera.position = Vector3(-0.126737f, 0.613379f, 3.716630f);
		camera.rotation = Quaternion(-0.107255f, -0.002421f, 0.000262f, -0.994227f);
	} else if (strcmp(scene_name, DATA_PATH("cornellbox.obj")) == 0) {
		camera.position = Vector3(0.528027f, 1.004323f, -0.774033f);
		camera.rotation = Quaternion(0.035059f, -0.963870f, 0.208413f, 0.162142f);
	} else if (strcmp(scene_name, DATA_PATH("glossy.obj")) == 0) {
		camera.position = Vector3(9.467193f, 5.919240f, 0.646071f);
		camera.rotation = Quaternion(0.179088f, -0.677310f, 0.175366f, 0.691683f);
	} else {
		camera.position = Vector3(1.272743f, 3.097532f, -3.189943f);
		camera.rotation = Quaternion(0.000000f, 0.995683f, 0.000000f, -0.092814f);
	}
}

void Pathtracer::update(float delta, const unsigned char * keys) {
	camera.update(delta, keys);

	//if (camera.moved) {
	//	frames_since_camera_moved = 0;
	//} else {
	//	frames_since_camera_moved++;
	//}

	frames_since_camera_moved = (frames_since_camera_moved + 1) & 255;
}

void Pathtracer::render() {
	if (camera.rasterize) {
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

		shader.unbind();
		gbuffer.unbind();
	
		// Sync Async Memcpy stream
		CUDACALL(cuStreamSynchronize(memcpy_stream));
	
		glFinish();

		// Convert rasterized GBuffers into primary Rays
		kernel_primary.execute(
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
			camera.position, 
			camera.bottom_left_corner_rotated, 
			camera.x_axis_rotated, 
			camera.y_axis_rotated
		);

		CUDACALL(cuStreamSynchronize(memcpy_stream));

		kernel_extend.execute(rand(), 0);
	}

	// Process the various Material types in different Kernels
	kernel_shade_diffuse.execute   (rand(), 0, frames_since_camera_moved);
	kernel_shade_dielectric.execute(rand(), 0);
	kernel_shade_glossy.execute    (rand(), 0, frames_since_camera_moved);

	// Trace shadow Rays
	kernel_connect.execute(rand(), 0, frames_since_camera_moved);

	for (int bounce = 1; bounce < NUM_BOUNCES; bounce++) {
		// Extend all Rays that are still alive to their next Triangle intersection
		kernel_extend.execute(rand(), bounce);

		// Process the various Material types in different Kernels
		kernel_shade_diffuse.execute   (rand(), bounce, frames_since_camera_moved);
		kernel_shade_dielectric.execute(rand(), bounce);
		kernel_shade_glossy.execute    (rand(), bounce, frames_since_camera_moved);

		// Trace shadow Rays
		kernel_connect.execute(rand(), bounce, frames_since_camera_moved);
	}

	// Reset buffer sizes to default for next frame
	global_buffer_sizes.set_value_async(buffer_sizes, memcpy_stream);

	// Integrate temporally
	kernel_temporal.execute();

	// À-Trous Filter
	const int atrous_iterations = 5;

	CUdeviceptr direct_in    = ptr_direct.ptr;
	CUdeviceptr indirect_in  = ptr_indirect.ptr;
	CUdeviceptr direct_out   = ptr_direct_alt.ptr;
	CUdeviceptr indirect_out = ptr_indirect_alt.ptr;

	for (int i = 0; i < atrous_iterations; i++) {
		int step_size = 1 << i;

		kernel_atrous.execute(direct_in, indirect_in, direct_out, indirect_out, step_size);
		
		// Ping-Pong the Frame Buffers
		std::swap(direct_in,   direct_out);
		std::swap(indirect_in, indirect_out);
	}

	//if constexpr (atrous_iterations == 0) {
	//	static CUdeviceptr history_direct   = module.get_global("history_direct").ptr;
	//	static CUdeviceptr history_indirect = module.get_global("history_indirect").ptr;

	//	CUDACALL(cuMemcpyDtoD(history_direct,   ptr_direct.ptr,   SCREEN_WIDTH * SCREEN_HEIGHT * 4 * sizeof(float)));
	//	CUDACALL(cuMemcpyDtoD(history_indirect, ptr_indirect.ptr, SCREEN_WIDTH * SCREEN_HEIGHT * 4 * sizeof(float)));
	//}
	
	std::swap(direct_in,   direct_out);
	std::swap(indirect_in, indirect_out);

	kernel_finalize.execute(direct_out, indirect_out);

	// Sync Main stream
	CUDACALL(cuStreamSynchronize(nullptr));
}
