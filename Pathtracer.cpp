#include "Pathtracer.h"

#include "CUDAMemory.h"
#include "CUDAContext.h"

#include "MeshData.h"
#include "BVH.h"

#include "Sky.h"

#include "ScopedTimer.h"

const int PIXEL_COUNT = SCREEN_WIDTH * SCREEN_HEIGHT;

void Pathtracer::init(unsigned frame_buffer_handle) {
	CUDAContext::init();

	camera.init(DEG_TO_RAD(110.0f));
	camera.resize(SCREEN_WIDTH, SCREEN_HEIGHT);

	// Init CUDA Module and its Kernel
	CUDAModule module;
	module.init("CUDA_Source/Pathtracer.cu", CUDAContext::compute_capability);

	const char * scene_name = DATA_PATH("scene.obj");
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

	Vector3 * triangles_position0      = new Vector3[bvh.primitive_count];
	Vector3 * triangles_position_edge1 = new Vector3[bvh.primitive_count];
	Vector3 * triangles_position_edge2 = new Vector3[bvh.primitive_count];

	Vector3 * triangles_normal0      = new Vector3[bvh.primitive_count];
	Vector3 * triangles_normal_edge1 = new Vector3[bvh.primitive_count];
	Vector3 * triangles_normal_edge2 = new Vector3[bvh.primitive_count]; 
	
	alignas(8) Vector2 * triangles_tex_coord0      = new Vector2[bvh.primitive_count];
	alignas(8) Vector2 * triangles_tex_coord_edge1 = new Vector2[bvh.primitive_count];
	alignas(8) Vector2 * triangles_tex_coord_edge2 = new Vector2[bvh.primitive_count];

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
	module.set_surface("frame_buffer", CUDAContext::map_gl_texture(frame_buffer_handle));

	struct PathBuffer {
		CUDAMemory::Ptr<Vector3> origin;
		CUDAMemory::Ptr<Vector3> direction;
	
		CUDAMemory::Ptr<int> triangle_id;
		CUDAMemory::Ptr<float> u;
		CUDAMemory::Ptr<float> v;
		CUDAMemory::Ptr<float> t;

		CUDAMemory::Ptr<int> pixel_index;
		CUDAMemory::Ptr<Vector3> colour;
		CUDAMemory::Ptr<Vector3> throughput;

		inline void init(int buffer_size) {
			origin    = CUDAMemory::malloc<Vector3>(buffer_size);
			direction = CUDAMemory::malloc<Vector3>(buffer_size);

			triangle_id = CUDAMemory::malloc<int>(buffer_size);
			u = CUDAMemory::malloc<float>(buffer_size);
			v = CUDAMemory::malloc<float>(buffer_size);
			t = CUDAMemory::malloc<float>(buffer_size);

			pixel_index = CUDAMemory::malloc<int>(buffer_size);
			colour     = CUDAMemory::malloc<Vector3>(buffer_size);
			throughput = CUDAMemory::malloc<Vector3>(buffer_size);
		}
	};

	PathBuffer buffer_0;
	PathBuffer buffer_1;
	
	int pixel_count = SCREEN_WIDTH * SCREEN_HEIGHT;
	buffer_0.init(pixel_count);
	buffer_1.init(pixel_count);

	global_buffer_0 = module.get_global("buffer_0");
	global_buffer_1 = module.get_global("buffer_1");

	global_buffer_0.set_value(buffer_0);
	global_buffer_1.set_value(buffer_1);

	global_N_ext = module.get_global("N_ext");

	kernel_generate.init(&module, "kernel_generate");
	kernel_extend.init  (&module, "kernel_extend");
	kernel_shade.init   (&module, "kernel_shade");
	kernel_connect.init (&module, "kernel_connect");

	kernel_generate.set_block_dim(128, 1, 1);
	kernel_extend.set_block_dim  (128, 1, 1);
	kernel_shade.set_block_dim   (128, 1, 1);
	kernel_connect.set_block_dim (128, 1, 1);

	kernel_generate.set_grid_dim(PIXEL_COUNT / kernel_generate.block_dim_x, 1, 1);
	kernel_extend.set_grid_dim  (PIXEL_COUNT / kernel_extend.block_dim_x,   1, 1);
	kernel_shade.set_grid_dim   (PIXEL_COUNT / kernel_shade.block_dim_x,    1, 1);
	kernel_connect.set_grid_dim (PIXEL_COUNT / kernel_connect.block_dim_x,  1, 1);

	if (strcmp(scene_name, DATA_PATH("pica/pica.obj")) == 0) {
		camera.position = Vector3(-14.875896f, 5.407789f, 22.486183f);
		camera.rotation = Quaternion(0.000000f, 0.980876f, 0.000000f, 0.194635f);
	} else if (strcmp(scene_name, DATA_PATH("sponza/sponza.obj")) == 0) {
		camera.position = Vector3(2.698714f, 39.508224f, 15.633610f);
		camera.rotation = Quaternion(0.000000f, -0.891950f, 0.000000f, 0.452135f);
	} else if (strcmp(scene_name, DATA_PATH("scene.obj")) == 0) {
		camera.position = Vector3(-0.101589f, 0.613379f, 3.580916f);
		camera.rotation = Quaternion(-0.006744f, 0.992265f, -0.107043f, -0.062512f);
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

	kernel_generate.execute(
		rand(), 
		PIXEL_COUNT,
		camera.position, 
		camera.top_left_corner_rotated, 
		camera.x_axis_rotated, 
		camera.y_axis_rotated
	);

	CUdeviceptr ray_buffers[2] = { 
		global_buffer_0.ptr, 
		global_buffer_1.ptr
	};

	int alive_paths = PIXEL_COUNT;

	const int NUM_BOUNCES = 5;
	for (int bounce = 0; bounce < NUM_BOUNCES; bounce++) {
		int buffer_index = bounce & 1;
		const CUdeviceptr & ray_buffer_in  = ray_buffers[    buffer_index];
		const CUdeviceptr & ray_buffer_out = ray_buffers[1 - buffer_index];

		global_N_ext.set_value(0);

		kernel_extend.execute(alive_paths, ray_buffer_in);

		kernel_shade.execute(rand(), alive_paths, bounce, frames_since_camera_moved, ray_buffer_in, ray_buffer_out);

		alive_paths = global_N_ext.get_value<int>();
		if (alive_paths == 0) break;
	}
}
