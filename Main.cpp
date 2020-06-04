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

// Index of frame to take screen capture on
static constexpr int capture_frame_index = 0;


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
		const unsigned char * keys = SDL_GetKeyboardState(nullptr);

		pathtracer.update(delta_time, keys);
		pathtracer.render();
		
		window.draw_quad();
		
		if (keys[SDL_SCANCODE_P] || current_frame == capture_frame_index) {
			char screenshot_name[32];
			sprintf_s(screenshot_name, "screenshot_%i.ppm", current_frame);

			capture_screen(window, screenshot_name);
		}
		
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

		// Draw GUI
		window.gui_begin();

		ImGui::Begin("Pathtracer");

		if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Frame: %i - Index: %i", current_frame, pathtracer.frames_since_camera_moved);
			ImGui::Text("Delta: %.2f ms", 1000.0f * delta_time);
			ImGui::Text("Avg:   %.2f ms", 1000.0f * avg);
			ImGui::Text("FPS: %i", fps);
			
			ImGui::BeginChild("Performance Region", ImVec2(0, 150), true);

			bool category_changed = true;
			int  padding;

			// Display Profile timings per category
			for (int i = 0; i < pathtracer.events.size() - 1; i++) {
				if (category_changed) {
					padding = 0;

					// Sum the times of all events in the new Category so it can be displayed in the header
					float time_sum = 0.0f;

					int j;
					for (j = i; j < pathtracer.events.size() - 1; j++) {
						int length = strlen(pathtracer.events[j]->name);
						if (length > padding) padding = length;

						time_sum += CUDAEvent::time_elapsed_between(*pathtracer.events[j], *pathtracer.events[j + 1]);

						if (strcmp(pathtracer.events[j]->category, pathtracer.events[j + 1]->category) != 0) break;
					}

					bool category_visible = ImGui::TreeNode(pathtracer.events[i]->category, "%s: %.2f ms", pathtracer.events[i]->category, time_sum);
					if (!category_visible) {
						// Skip ahead to next category
						i = j;

						continue;
					}
				}

				const CUDAEvent * event_curr = pathtracer.events[i];
				const CUDAEvent * event_next = pathtracer.events[i + 1];

				float time = CUDAEvent::time_elapsed_between(*event_curr, *event_next);

				ImGui::Text("%s: %*.2f ms", event_curr->name, 5 + padding - strlen(event_curr->name), time);

				category_changed = strcmp(pathtracer.events[i]->category, pathtracer.events[i + 1]->category);
				if (category_changed) {
					ImGui::TreePop();
				}
			}

			ImGui::EndChild();
		}

		if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
			bool settings_changed = false;

			settings_changed |= ImGui::Checkbox("Rasterize Primary Rays",  &pathtracer.enable_rasterization);
			settings_changed |= ImGui::Checkbox("SVGF",                    &pathtracer.enable_svgf);
			settings_changed |= ImGui::Checkbox("Spatial Variance",        &pathtracer.enable_spatial_variance);
			settings_changed |= ImGui::Checkbox("TAA",                     &pathtracer.enable_taa);
			settings_changed |= ImGui::Checkbox("Modulate Albedo",         &pathtracer.enable_albedo);

			settings_changed |= ImGui::SliderInt("A Trous iterations", &pathtracer.svgf_settings.atrous_iterations, 0, MAX_ATROUS_ITERATIONS);

			settings_changed |= ImGui::SliderFloat("Alpha colour", &pathtracer.svgf_settings.alpha_colour, 0.0f, 1.0f);
			settings_changed |= ImGui::SliderFloat("Alpha moment", &pathtracer.svgf_settings.alpha_moment, 0.0f, 1.0f);

			pathtracer.settings_changed = settings_changed;
		}

		ImGui::End();

		window.gui_end();
		window.swap();
	}

	return EXIT_SUCCESS;
}
