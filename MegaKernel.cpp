#include "MegaKernel.h"

#include "CUDAMemory.h"
#include "CUDAContext.h"

void MegaKernel::init(const char * scene_name, const char * sky_name, unsigned frame_buffer_handle) {
	Pathtracer::init("CUDA_Source/megakernel.cu", scene_name, sky_name);

	// Set frame buffer to a CUDA resource mapping of the GL frame buffer texture
	module.set_surface("frame_buffer", CUDAContext::map_gl_texture(frame_buffer_handle));

	// Set Camera globals
	global_camera_position        = module.get_global("camera_position");
	global_camera_top_left_corner = module.get_global("camera_top_left_corner");
	global_camera_x_axis          = module.get_global("camera_x_axis");
	global_camera_y_axis          = module.get_global("camera_y_axis");

	kernel.init(&module, "trace_ray");

	kernel.set_block_dim(32, 4, 1);
	kernel.set_grid_dim(SCREEN_WIDTH / kernel.block_dim_x, SCREEN_HEIGHT / kernel.block_dim_y, 1);
}

void MegaKernel::render() {
	global_camera_position.set_value(camera.position);
	global_camera_top_left_corner.set_value(camera.top_left_corner_rotated);
	global_camera_x_axis.set_value(camera.x_axis_rotated);
	global_camera_y_axis.set_value(camera.y_axis_rotated);

	kernel.execute(rand(), frames_since_camera_moved);
}
