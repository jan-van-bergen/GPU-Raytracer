#include "Wavefront.h"

#include "CUDAMemory.h"
#include "CUDAContext.h"

void Wavefront::init(const char * scene_name, unsigned frame_buffer_handle) {
	Pathtracer::init("CUDA_Source/wavefront.cu", scene_name);

	// Set frame buffer to a CUDA resource mapping of the GL frame buffer texture
	module.set_surface("frame_buffer", CUDAMemory::create_array3d(SCREEN_WIDTH, SCREEN_HEIGHT, 1, 4, CUarray_format::CU_AD_FORMAT_FLOAT, CUDA_ARRAY3D_SURFACE_LDST));
	module.set_surface("accumulator", CUDAContext::map_gl_texture(frame_buffer_handle));

	struct ExtendBuffer {
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

	struct MaterialBuffer {
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

	module.get_global("ray_buffer_extend").set_value          (ray_buffer_extend);
	module.get_global("ray_buffer_shade_diffuse").set_value   (ray_buffer_shade_diffuse);
	module.get_global("ray_buffer_shade_dielectric").set_value(ray_buffer_shade_dielectric);
	module.get_global("ray_buffer_shade_glossy").set_value    (ray_buffer_shade_glossy);
	module.get_global("ray_buffer_connect").set_value         (ray_buffer_connect);

	global_N_extend     = module.get_global("N_extend");
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

void Wavefront::update(float delta, const unsigned char * keys) {
	camera.update(delta, keys);
}

void Wavefront::render() {
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

	global_N_extend.set_value(PIXEL_COUNT);

	for (int bounce = 0; bounce < NUM_BOUNCES; bounce++) {
		// Extend all Rays that are still alive to their next Triangle intersection
		kernel_extend.execute(rand());
		global_N_extend.set_value(0);

		// Process the various Material types in different Kernels
		kernel_shade_diffuse.execute(rand());
		kernel_shade_dielectric.execute(rand());
		kernel_shade_glossy.execute(rand());

		// Trace shadow Rays
		kernel_connect.execute(rand());

		global_N_diffuse.set_value(0);
		global_N_dielectric.set_value(0);
		global_N_glossy.set_value(0);
		global_N_shadow.set_value(0);
	}

	kernel_accumulate.execute(frames_since_camera_moved);
}
