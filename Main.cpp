#include <cstdio>
#include <cstdlib>
#include <time.h> 

#include <Imgui/imgui.h>

#include "Window.h"

#include "ScopedTimer.h"

#include "Pathtracer.h"

// Forces NVIDIA driver to be used 
extern "C" { _declspec(dllexport) unsigned NvOptimusEnablement = true; }

static void capture_screen(const Window & window, const char * file_name) {
	ScopedTimer timer("Screenshot");

	unsigned char * data = new unsigned char[SCREEN_WIDTH * SCREEN_HEIGHT * 3];
	unsigned char * temp = new unsigned char[SCREEN_WIDTH * 3];
			
	window.read_frame_buffer(data);

	// Flip image vertically
	for (int j = 0; j < SCREEN_HEIGHT / 2; j++) {
		unsigned char * row_top    = data +                  j      * SCREEN_WIDTH * 3;
		unsigned char * row_bottom = data + (SCREEN_HEIGHT - j - 1) * SCREEN_WIDTH * 3;

		memcpy(temp,       row_top,    SCREEN_WIDTH * 3);
		memcpy(row_top,    row_bottom, SCREEN_WIDTH * 3);
		memcpy(row_bottom, temp,       SCREEN_WIDTH * 3);
	}

	Util::export_ppm(file_name, SCREEN_WIDTH, SCREEN_HEIGHT, data);

	delete [] temp;
	delete [] data;
}

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
		avg /= float(count);

		// Calculate fps
		frames++;

		second += delta_time;
		while (second >= 1.0f) {
			second -= 1.0f;

			fps = frames;
			frames = 0;
		}

		window.draw_quad();
		
		if (current_frame == -1) {
			capture_screen(window, "debug.ppm");
		}
		
		window.gui_begin();

		ImGui::Begin("Pathtracer");

		if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Frame: %i - Index: %i", current_frame, pathtracer.frames_since_camera_moved);
			ImGui::Text("Delta: %.2f ms", 1000.0f * delta_time);
			ImGui::Text("Avg:   %.2f ms", 1000.0f * avg);
			ImGui::Text("FPS: %i", fps);
			
			ImGui::Text(" - Primary: %.2f ms", pathtracer.time_primary);

			float time_bounce[NUM_BOUNCES];

			float sum_bounce = 0.0f;
			float sum_atrous = 0.0f;

			for (int i = 0; i < NUM_BOUNCES; i++) {
				time_bounce[i] = 
					pathtracer.time_extend[i] + 
					pathtracer.time_shade_diffuse[i] + 
					pathtracer.time_shade_dielectric[i] + 
					pathtracer.time_shade_glossy[i] + 
					pathtracer.time_connect[i];
				sum_bounce += time_bounce[i];
			}

			for (int i = 0; i < ATROUS_ITERATIONS; i++) {
				sum_atrous += pathtracer.time_svgf_atrous[i];
			}

			if (ImGui::TreeNode("Bounces", "Bounces: %.2f ms", sum_bounce)) {
				char str_id[16];

				for (int i = 0; i < NUM_BOUNCES; i++) {	
					sprintf_s(str_id, "Bounce %i", i);

					if (ImGui::TreeNode(str_id, "%i: %.2f ms", i, time_bounce[i])) {
						ImGui::Text("Extend:     %.2f ms", pathtracer.time_extend[i]);
						ImGui::Text("Diffuse:    %.2f ms", pathtracer.time_shade_diffuse[i]);
						ImGui::Text("Dielectric: %.2f ms", pathtracer.time_shade_dielectric[i]);
						ImGui::Text("Glossy:     %.2f ms", pathtracer.time_shade_glossy[i]);
						ImGui::Text("Connect:    %.2f ms", pathtracer.time_connect[i]);

						ImGui::TreePop();
					}
				}
				
				ImGui::TreePop();
			}

			ImGui::Text(" - SVGF Temporal: %.2f ms", pathtracer.time_svgf_temporal);

			if (ImGui::TreeNode("Atrous", "SVGF atrous: %.2f ms", sum_atrous)) {
				for (int i = 0; i < ATROUS_ITERATIONS; i++) {
					ImGui::Text("%i: %.2f ms", i, pathtracer.time_svgf_atrous[i]);
				}

				ImGui::TreePop();
			}

			ImGui::Text(" - SVGF Finalize: %.2f ms", pathtracer.time_svgf_finalize);

			ImGui::Text(" - TAA: %.2f ms", pathtracer.time_taa);
		}

		if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
			bool settings_changed = false;

			settings_changed |= ImGui::Checkbox("Rasterize Primary Rays", &pathtracer.enable_rasterization);
			settings_changed |= ImGui::Checkbox("Enable SVGF",            &pathtracer.enable_svgf);
			settings_changed |= ImGui::Checkbox("Enable TAA",             &pathtracer.enable_taa);
			settings_changed |= ImGui::Checkbox("Modulate Albedo",        &pathtracer.enable_albedo);

			settings_changed |= ImGui::SliderFloat("alpha colour", &pathtracer.svgf_settings.alpha_colour, 0.0f, 1.0f);
			settings_changed |= ImGui::SliderFloat("alpha moment", &pathtracer.svgf_settings.alpha_moment, 0.0f, 1.0f);

			pathtracer.settings_changed = settings_changed;
		}

		ImGui::End();

		window.gui_end();
		window.swap();
	}

	return EXIT_SUCCESS;
}
