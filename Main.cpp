#include <cstdio>
#include <cstdlib>

#include "Window.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"

#include "Camera.h"

#include "MeshData.h"

// Forces NVIDIA driver to be used 
extern "C" { _declspec(dllexport) unsigned NvOptimusEnablement = true; }

#define TOTAL_TIMING_COUNT 1000
float timings[TOTAL_TIMING_COUNT];
int   current_frame = 0;

int main(int argument_count, char ** arguments) {
	Window window(SCREEN_WIDTH, SCREEN_HEIGHT, "Pathtracer");

	// Initialize timing stuff
	Uint64 now  = 0;
	Uint64 last = 0;
	float inv_perf_freq = 1.0f / (float)SDL_GetPerformanceFrequency();
	float delta_time = 0;

	float second = 0.0f;
	int frames = 0;
	int fps    = 0;

	Camera camera(DEG_TO_RAD(110.0f));
	camera.resize(SCREEN_WIDTH, SCREEN_HEIGHT);

	// Init CUDA Module and its Kernel
	CUDAModule module;
	module.init("test.cu", window.cuda_compute_capability);

	const MeshData * mesh = MeshData::load(DATA_PATH("Torus.obj"));

	CUdeviceptr ptr;
	CUDACALL(cuMemAlloc(&ptr, mesh->triangle_count * sizeof(Triangle)));

	CUDACALL(cuMemcpyHtoD(ptr, mesh->triangles, mesh->triangle_count * sizeof(Triangle)));

	module.get_global("triangle_count").set(mesh->triangle_count);
	module.get_global("triangles").set(ptr);

	CUDAModule::Global global_camera_position        = module.get_global("camera_position");
	CUDAModule::Global global_camera_top_left_corner = module.get_global("camera_top_left_corner");
	CUDAModule::Global global_camera_x_axis          = module.get_global("camera_x_axis");
	CUDAModule::Global global_camera_y_axis          = module.get_global("camera_y_axis");

	CUDAKernel kernel;
	kernel.init(&module, "trace_ray");

	kernel.set_block_dim(32, 4, 1);
	kernel.set_grid_dim(SCREEN_WIDTH / kernel.block_dim_x, SCREEN_HEIGHT / kernel.block_dim_y, 1);

	kernel.set_surface("output_surface", window.cuda_frame_buffer);

	last = SDL_GetPerformanceCounter();

	// Game loop
	while (!window.is_closed) {
		window.clear();
		
		camera.update(delta_time, SDL_GetKeyboardState(NULL));
		
		global_camera_position.set(camera.position);
		global_camera_top_left_corner.set(camera.top_left_corner_rotated);
		global_camera_x_axis.set(camera.x_axis_rotated);
		global_camera_y_axis.set(camera.y_axis_rotated);

		kernel.execute();

		window.update();

		// Perform frame timing
		now = SDL_GetPerformanceCounter();
		delta_time = float(now - last) * inv_perf_freq;
		last = now;

		// Calculate average of last TOTAL_TIMING_COUNT frames
		timings[current_frame++ % TOTAL_TIMING_COUNT] = delta_time;

		float avg = 0.0f;
		int count = current_frame < TOTAL_TIMING_COUNT ? current_frame : TOTAL_TIMING_COUNT;
		for (int i = 0; i < count; i++) {
			avg += timings[i];
		}
		avg /= count;

		// Calculate fps
		frames++;

		second += delta_time;
		while (second >= 1.0f) {
			second -= 1.0f;

			fps = frames;
			frames = 0;
		}

		// Report timings
		printf("%d - Delta: %.2f ms, Average: %.2f ms, FPS: %d        \r", current_frame, delta_time * 1000.0f, avg * 1000.0f, fps);
	}

	return EXIT_SUCCESS;
}
