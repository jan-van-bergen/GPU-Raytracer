#include "Pathtracer.h"

#include <filesystem>

#include "CUDAMemory.h"
#include "CUDAContext.h"

#include "MeshData.h"
#include "BVH.h"

#include "Sky.h"

#include "ScopedTimer.h"

const int PIXEL_COUNT = SCREEN_WIDTH * SCREEN_HEIGHT;

void Pathtracer::init(const char * scene_name, unsigned frame_buffer_handle) {
	CUDAContext::init();

	camera.init(DEG_TO_RAD(110.0f));
	camera.resize(SCREEN_WIDTH, SCREEN_HEIGHT);

	// Init CUDA Module and its Kernel
	CUDAModule module;
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
			tex_desc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
			tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
			
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
				bvh.primitives[i].position0, 
				bvh.primitives[i].position1, 
				bvh.primitives[i].position2
			};
			bvh.primitives[i].aabb = AABB::from_points(vertices, 3);
		}

		{
			ScopedTimer timer("BVH Construction");

			bvh.build_sbvh();
		}

		bvh.save_to_disk(bvh_filename.c_str());
	}

	// Allocate Triangles in SoA format
	Vector3 * triangles_position0      = new Vector3[bvh.primitive_count];
	Vector3 * triangles_position_edge1 = new Vector3[bvh.primitive_count];
	Vector3 * triangles_position_edge2 = new Vector3[bvh.primitive_count];

	Vector3 * triangles_normal0      = new Vector3[bvh.primitive_count];
	Vector3 * triangles_normal_edge1 = new Vector3[bvh.primitive_count];
	Vector3 * triangles_normal_edge2 = new Vector3[bvh.primitive_count]; 
	
	Vector2 * triangles_tex_coord0      = new Vector2[bvh.primitive_count];
	Vector2 * triangles_tex_coord_edge1 = new Vector2[bvh.primitive_count];
	Vector2 * triangles_tex_coord_edge2 = new Vector2[bvh.primitive_count];

	int * triangles_material_id = new int[bvh.primitive_count];

	for (int i = 0; i < bvh.primitive_count; i++) {
		triangles_position0[i]      = bvh.primitives[i].position0;
		triangles_position_edge1[i] = bvh.primitives[i].position1 - bvh.primitives[i].position0;
		triangles_position_edge2[i] = bvh.primitives[i].position2 - bvh.primitives[i].position0;

		triangles_normal0[i]      = bvh.primitives[i].normal0;
		triangles_normal_edge1[i] = bvh.primitives[i].normal1 - bvh.primitives[i].normal0;
		triangles_normal_edge2[i] = bvh.primitives[i].normal2 - bvh.primitives[i].normal0;

		triangles_tex_coord0[i]      = bvh.primitives[i].tex_coord0;
		triangles_tex_coord_edge1[i] = bvh.primitives[i].tex_coord1 - bvh.primitives[i].tex_coord0;
		triangles_tex_coord_edge2[i] = bvh.primitives[i].tex_coord2 - bvh.primitives[i].tex_coord0;

		triangles_material_id[i] = bvh.primitives[i].material_id;
	}

	// Set global Triangle buffers
	module.get_global("triangles_position0"     ).set_buffer(triangles_position0,      bvh.primitive_count);
	module.get_global("triangles_position_edge1").set_buffer(triangles_position_edge1, bvh.primitive_count);
	module.get_global("triangles_position_edge2").set_buffer(triangles_position_edge2, bvh.primitive_count);

	module.get_global("triangles_normal0"     ).set_buffer(triangles_normal0,      bvh.primitive_count);
	module.get_global("triangles_normal_edge1").set_buffer(triangles_normal_edge1, bvh.primitive_count);
	module.get_global("triangles_normal_edge2").set_buffer(triangles_normal_edge2, bvh.primitive_count);

	module.get_global("triangles_tex_coord0"     ).set_buffer(triangles_tex_coord0,      bvh.primitive_count);
	module.get_global("triangles_tex_coord_edge1").set_buffer(triangles_tex_coord_edge1, bvh.primitive_count);
	module.get_global("triangles_tex_coord_edge2").set_buffer(triangles_tex_coord_edge2, bvh.primitive_count);

	module.get_global("triangles_material_id").set_buffer(triangles_material_id, bvh.primitive_count);

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

	int * light_indices = new int[mesh->triangle_count];
	int   light_count = 0;

	for (int i = 0; i < mesh->triangle_count; i++) {
		const Triangle & triangle = mesh->triangles[i];

		if (Vector3::length_squared(mesh->materials[triangle.material_id].emittance) > 0.0f) {
			int index = -1;
			for (int j = 0; j < bvh.primitive_count; j++) {
				if (bvh.indices_x[j] == i) {
					index = j;

					break;
				}
			}

			light_indices[light_count++] = index;
		}
	}
	
	if (light_count > 0) {
		module.get_global("light_indices").set_buffer(light_indices, light_count);
	}

	delete [] light_indices;

	module.get_global("light_count").set_value(light_count);

	// Set global BVHNode buffer
	module.get_global("bvh_nodes").set_buffer(bvh.nodes, bvh.node_count);

	// Set Sky globals
	Sky sky;
	sky.init(DATA_PATH("Sky_Probes/rnl_probe.float"));

	module.get_global("sky_size").set_value(sky.size);
	module.get_global("sky_data").set_buffer(sky.data, sky.size * sky.size);

	// Set frame buffer to a CUDA resource mapping of the GL frame buffer texture
	module.set_surface("frame_buffer", CUDAMemory::create_array3d(SCREEN_WIDTH, SCREEN_HEIGHT, 1, 4, CUarray_format::CU_AD_FORMAT_FLOAT, CUDA_ARRAY3D_SURFACE_LDST));
	module.set_surface("accumulator", CUDAContext::map_gl_texture(frame_buffer_handle));

	struct RayBuffer {
		CUDAMemory::Ptr<float> origin_x;
		CUDAMemory::Ptr<float> origin_y;
		CUDAMemory::Ptr<float> origin_z;
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

		CUDAMemory::Ptr<bool> last_specular;

		inline void init(int buffer_size) {
			origin_x    = CUDAMemory::malloc<float>(buffer_size);
			origin_y    = CUDAMemory::malloc<float>(buffer_size);
			origin_z    = CUDAMemory::malloc<float>(buffer_size);
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

			last_specular = CUDAMemory::malloc<bool>(buffer_size);
		}
	};

	RayBuffer ray_buffer_extend;
	RayBuffer ray_buffer_shade_diffuse;
	RayBuffer ray_buffer_shade_dielectric;
	RayBuffer ray_buffer_shade_glossy;
	
	ray_buffer_extend.init          (PIXEL_COUNT);
	ray_buffer_shade_diffuse.init   (PIXEL_COUNT);
	ray_buffer_shade_dielectric.init(PIXEL_COUNT);
	ray_buffer_shade_glossy.init    (PIXEL_COUNT);

	module.get_global("ray_buffer_extend").set_value          (ray_buffer_extend);
	module.get_global("ray_buffer_shade_diffuse").set_value   (ray_buffer_shade_diffuse);
	module.get_global("ray_buffer_shade_dielectric").set_value(ray_buffer_shade_dielectric);
	module.get_global("ray_buffer_shade_glossy").set_value    (ray_buffer_shade_glossy);

	struct ShadowRayBuffer {
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

	ShadowRayBuffer shadow_ray_buffer;
	shadow_ray_buffer.init(PIXEL_COUNT);

	module.get_global("shadow_ray_buffer").set_value(shadow_ray_buffer);

	global_N_ext        = module.get_global("N_ext");
	global_N_diffuse    = module.get_global("N_diffuse");
	global_N_dielectric = module.get_global("N_dielectric");
	global_N_glossy     = module.get_global("N_glossy");
	global_N_shadow     = module.get_global("N_shadow");
	
	global_N_diffuse.set_value   (0);
	global_N_dielectric.set_value(0);
	global_N_glossy.set_value    (0);
	global_N_shadow.set_value    (0);

	kernel_generate.init        (&module, "kernel_generate");
	kernel_extend.init          (&module, "kernel_extend");
	kernel_shade_diffuse.init   (&module, "kernel_shade_diffuse");
	kernel_shade_dielectric.init(&module, "kernel_shade_dielectric");
	kernel_shade_glossy.init    (&module, "kernel_shade_glossy");
	kernel_connect.init         (&module, "kernel_connect");
	kernel_accumulate.init      (&module, "kernel_accumulate");

	kernel_generate.set_block_dim        (256, 1, 1);
	kernel_extend.set_block_dim          (256, 1, 1);
	kernel_shade_diffuse.set_block_dim   (256, 1, 1);
	kernel_shade_dielectric.set_block_dim(256, 1, 1);
	kernel_shade_glossy.set_block_dim    (256, 1, 1);
	kernel_connect.set_block_dim         (256, 1, 1);
	kernel_accumulate.set_block_dim(32, 4, 1);

	kernel_generate.set_grid_dim        (PIXEL_COUNT / kernel_generate.block_dim_x,         1, 1);
	kernel_extend.set_grid_dim          (PIXEL_COUNT / kernel_extend.block_dim_x,           1, 1);
	kernel_shade_diffuse.set_grid_dim   (PIXEL_COUNT / kernel_shade_diffuse.block_dim_x,    1, 1);
	kernel_shade_dielectric.set_grid_dim(PIXEL_COUNT / kernel_shade_dielectric.block_dim_x, 1, 1);
	kernel_shade_glossy.set_grid_dim    (PIXEL_COUNT / kernel_shade_glossy.block_dim_x,     1, 1);
	kernel_connect.set_grid_dim         (PIXEL_COUNT / kernel_connect.block_dim_x,          1, 1);
	kernel_accumulate.set_grid_dim(
		SCREEN_WIDTH  / kernel_accumulate.block_dim_x, 
		SCREEN_HEIGHT / kernel_accumulate.block_dim_y,
		1
	);

	if (strcmp(scene_name, DATA_PATH("pica/pica.obj")) == 0) {
		camera.position = Vector3(-14.875896f, 5.407789f, 22.486183f);
		camera.rotation = Quaternion(0.000000f, 0.980876f, 0.000000f, 0.194635f);
	} else if (strcmp(scene_name, DATA_PATH("sponza/sponza.obj")) == 0) {
		camera.position = Vector3(2.698714f, 39.508224f, 15.633610f);
		camera.rotation = Quaternion(0.000000f, -0.891950f, 0.000000f, 0.452135f);
	} else if (strcmp(scene_name, DATA_PATH("scene.obj")) == 0) {
		camera.position = Vector3(-0.101589f, 0.613379f, 3.580916f);
		camera.rotation = Quaternion(-0.006744f, 0.992265f, -0.107043f, -0.062512f);

		//camera.position = Vector3(-1.843730f, -0.213465f, -0.398855f);
		//camera.rotation = Quaternion(0.045417f, 0.900693f, -0.097165f, 0.421010f);

		//camera.position = Vector3(-1.526055f, 0.739711f, -1.135700f);
		//camera.rotation = Quaternion(-0.154179f, -0.830964f, 0.282224f, -0.453958f);
	} else if (strcmp(scene_name, DATA_PATH("cornellbox.obj")) == 0) {
		camera.position = Vector3(0.528027f, 1.004323f, 0.774033f);
		camera.rotation = Quaternion(0.035059f, -0.963870f, 0.208413f, 0.162142f);
	} else if (strcmp(scene_name, DATA_PATH("glossy.obj")) == 0) {
		camera.position = Vector3(9.467193f, 5.919240f, -0.646071f);
		camera.rotation = Quaternion(0.179088f, -0.677310f, 0.175366f, 0.691683f);
	}
}

void Pathtracer::update(float delta, const unsigned char * keys) {
	camera.update(delta, keys);
}

void Pathtracer::render() {
	if (camera.moved) {
		frames_since_camera_moved = 0.0f;
	} else {
		frames_since_camera_moved += 1.0f;
	}

	// Generate primary Rays from the current Camera orientation
	kernel_generate.execute(
		rand(),
		camera.position, 
		camera.top_left_corner_rotated, 
		camera.x_axis_rotated, 
		camera.y_axis_rotated
	);

	global_N_ext.set_value(PIXEL_COUNT);

	const int NUM_BOUNCES = 5;
	for (int bounce = 0; bounce < NUM_BOUNCES; bounce++) {
		// Extend all Rays that are still alive to their next Triangle intersection
		kernel_extend.execute(rand());
		global_N_ext.set_value(0);

		// Process the various Material types in different Kernels
		kernel_shade_diffuse.execute   (rand());
		kernel_shade_dielectric.execute(rand());
		kernel_shade_glossy.execute    (rand());

		// Trace shadow Rays
		kernel_connect.execute(rand());

		global_N_diffuse.set_value   (0);
		global_N_dielectric.set_value(0);
		global_N_glossy.set_value    (0);
		global_N_shadow.set_value    (0);
	}

	kernel_accumulate.execute(frames_since_camera_moved);
}
