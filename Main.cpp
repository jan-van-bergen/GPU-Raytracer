#include <cstdio>
#include <cstdlib>
#include <time.h> 

#include <Imgui/imgui.h>

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
	
	const char * scene_filename = DATA_PATH("sponza/sponza_lit.obj");
	const char * sky_filename   = DATA_PATH("Sky_Probes/rnl_probe.float");

	Pathtracer pathtracer;
	pathtracer.init(scene_filename, sky_filename, window.frame_buffer_handle);

	srand(1337);

	last = SDL_GetPerformanceCounter();

	// Game loop
	while (!window.is_closed) {
		pathtracer.update(delta_time, SDL_GetKeyboardState(nullptr));
		pathtracer.render();

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

		window.begin_gui();

		ImGui::Begin("Pathtracer");

		if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Frame: %i - Frames sample: %i", current_frame, pathtracer.frames_since_camera_moved);
			ImGui::Text("Delta: %.2f ms", delta_time * 1000.0f);
			ImGui::Text("Avg:   %.2f ms", avg        * 1000.0f);
			ImGui::Text("FPS: %i", fps);
			ImGui::Text("Primary: %.2f ms", pathtracer.time_primary);
			ImGui::Text("Extend:  %.2f ms", pathtracer.time_extend);
			ImGui::Text("SVGF:    %.2f ms", pathtracer.time_svgf);
		}

		if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Checkbox("Rasterize Primary Rays", &pathtracer.enable_rasterization);
			ImGui::Checkbox("Enable SVGF",            &pathtracer.enable_svgf);
			ImGui::Checkbox("Modulate Albedo",        &pathtracer.enable_albedo);

			ImGui::SliderFloat("simga z", &pathtracer.svgf_settings.sigma_z, 0.0f,  10.0f);
			ImGui::SliderFloat("simga n", &pathtracer.svgf_settings.sigma_n, 1.0f, 256.0f);
			ImGui::SliderFloat("sigma l", &pathtracer.svgf_settings.sigma_l, 4.0f, 400.0f);
		}

		ImGui::End();

		window.update();
	}

	return EXIT_SUCCESS;
}
