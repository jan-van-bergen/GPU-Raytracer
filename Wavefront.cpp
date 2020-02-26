#include "Wavefront.h"

#include "CUDAMemory.h"
#include "CUDAContext.h"

void Wavefront::init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle) {
	Pathtracer::init("CUDA_Source/wavefront.cu", scene_name, sky_name);

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

	kernel_generate.set_block_dim        (128, 1, 1);
	kernel_extend.set_block_dim          (128, 1, 1);
	kernel_shade_diffuse.set_block_dim   (128, 1, 1);
	kernel_shade_dielectric.set_block_dim(128, 1, 1);
	kernel_shade_glossy.set_block_dim    (128, 1, 1);
	kernel_connect.set_block_dim         (128, 1, 1);
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
}

void Wavefront::render() {
	// Generate primary Rays from the current Camera orientation
	kernel_generate.execute(
		rand(),
		frames_since_camera_moved,
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
		kernel_shade_diffuse.execute   (rand(), frames_since_camera_moved, bounce);
		kernel_shade_dielectric.execute(rand());
		kernel_shade_glossy.execute    (rand(), frames_since_camera_moved, bounce);

		// Trace shadow Rays
		kernel_connect.execute(rand(), frames_since_camera_moved, bounce);

		global_N_diffuse.set_value(0);
		global_N_dielectric.set_value(0);
		global_N_glossy.set_value(0);
		global_N_shadow.set_value(0);
	}

	kernel_accumulate.execute(float(frames_since_camera_moved));
}
