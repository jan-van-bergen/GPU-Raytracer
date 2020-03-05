#include <cstdio>
#include <cstdlib>
#include <time.h> 

#include "Window.h"

#include "ScopedTimer.h"

#include "Pathtracer.h"

// Forces NVIDIA driver to be used 
extern "C" { _declspec(dllexport) unsigned NvOptimusEnablement = true; }

#define TOTAL_TIMING_COUNT 100
float timings[TOTAL_TIMING_COUNT];
int   current_frame = 0;

int main(int argument_count, char ** arguments) {
	Window window("Pathtracer");
	
	// Initialize timing stuff
	Uint64 now  = 0;
	Uint64 last = 0;
	float inv_perf_freq = 1.0f / float(SDL_GetPerformanceFrequency());
	float delta_time = 0;

	float second = 0.0f;
	int frames = 0;
	int fps    = 0;
	
	const char * scene_filename = DATA_PATH("scene.obj");
	const char * sky_filename   = DATA_PATH("Sky_Probes/rnl_probe.float");

	Pathtracer pathtracer;
	pathtracer.init(scene_filename, sky_filename, window.frame_buffer_handle);

	srand(1337);

	last = SDL_GetPerformanceCounter();

	// Game loop
	while (!window.is_closed) {
		pathtracer.update(delta_time, SDL_GetKeyboardState(nullptr));
		pathtracer.render();

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
