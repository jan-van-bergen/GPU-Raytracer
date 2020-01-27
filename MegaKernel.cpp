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

void MegaKernel::render() {
	if (camera.moved) {
		frames_since_camera_moved = 0.0f;
	} else {
		frames_since_camera_moved += 1.0f;
	}
	
	global_camera_position.set_value(camera.position);
	global_camera_top_left_corner.set_value(camera.top_left_corner_rotated);
	global_camera_x_axis.set_value(camera.x_axis_rotated);
	global_camera_y_axis.set_value(camera.y_axis_rotated);

	kernel.execute(rand(), frames_since_camera_moved);
}
