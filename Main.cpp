#include <cstdio>

#include <Imgui/imgui.h>

#include "CUDAContext.h"

#include "Pathtracer.h"

#include "Input.h"
#include "Window.h"

#include "Random.h"

#include "Util.h"
#include "PerfTest.h"
#include "ScopeTimer.h"

// Forces NVIDIA driver to be used 
extern "C" { _declspec(dllexport) unsigned NvOptimusEnablement = true; }

static void capture_screen(const Window & window, const char * file_name) {
	ScopeTimer timer("Screenshot");

	unsigned char * data = new unsigned char[window.width * window.height * 3];
	unsigned char * temp = new unsigned char[window.width * 3];
			
	window.read_frame_buffer(data);

	// Flip image vertically
	for (int j = 0; j < window.height / 2; j++) {
		unsigned char * row_top    = data +                  j      * window.width * 3;
		unsigned char * row_bottom = data + (window.height - j - 1) * window.width * 3;

		memcpy(temp,       row_top,    window.width * 3);
		memcpy(row_top,    row_bottom, window.width * 3);
		memcpy(row_bottom, temp,       window.width * 3);
	}

	Util::export_ppm(file_name, window.width, window.height, data);

	delete [] temp;
	delete [] data;
}

#define FRAMETIME_HISTORY_LENGTH 100
static float frame_times[FRAMETIME_HISTORY_LENGTH];

static int current_frame = 0;

// Index of frame to take screen capture on
static constexpr int capture_frame_index = -1;
static constexpr bool exit_after_capture = true;

static Pathtracer pathtracer;
static PerfTest   perf_test;

static void window_resize(unsigned frame_buffer_handle, int width, int height) {
	pathtracer.resize_free();
	pathtracer.resize_init(frame_buffer_handle, width, height);
};

int main(int argument_count, char ** arguments) {
	Window window("Pathtracer");

	// Initialize timing stuff
	Uint64 now  = 0;
	Uint64 last = 0;
	float inv_perf_freq = 1.0f / float(SDL_GetPerformanceFrequency());
	float delta_time = 0;

	float second = 0.0f;
	int frames_this_second = 0;
	int fps = 0;
	
	const char * mesh_names[] = {
		DATA_PATH("sponza/sponza_lit.obj"),
	};
	const char * sky_filename = DATA_PATH("Sky_Probes/rnl_probe.float");

	pathtracer.init(Util::array_element_count(mesh_names), mesh_names, sky_filename, window.frame_buffer_handle);

	perf_test.init(&pathtracer, false, mesh_names[0]);

	window.resize_handler = &window_resize;

	Random::init(1337);

	last = SDL_GetPerformanceCounter();

	// Game loop
	while (!window.is_closed) {
		perf_test.frame_begin();

		pathtracer.update(delta_time);
		pathtracer.render();
		
		window.render_framebuffer();
		
		if (Input::is_key_pressed(SDL_SCANCODE_P) || current_frame == capture_frame_index) {
			char screenshot_name[32];
			sprintf_s(screenshot_name, "screenshot_%i.ppm", current_frame);

			capture_screen(window, screenshot_name);

			if (current_frame == capture_frame_index && exit_after_capture) break;
		}
		
		// Perform frame timing
		now = SDL_GetPerformanceCounter();
		delta_time = float(now - last) * inv_perf_freq;
		last = now;

		// Calculate average of last frames
		frame_times[current_frame++ % FRAMETIME_HISTORY_LENGTH] = delta_time;

		int count = current_frame < FRAMETIME_HISTORY_LENGTH ? current_frame : FRAMETIME_HISTORY_LENGTH;

		float avg = 0.0f;
		float min = INFINITY;
		float max = 0.0f;

		for (int i = 0; i < count; i++) {
			avg += frame_times[i];
			min = fminf(min, frame_times[i]);
			max = fmaxf(max, frame_times[i]);
		}
		avg /= float(count);

		// Calculate fps
		frames_this_second++;

		second += delta_time;
		while (second >= 1.0f) {
			second -= 1.0f;

			fps = frames_this_second;
			frames_this_second = 0;
		}

		// Draw GUI
		window.gui_begin();

		ImGui::Begin("Pathtracer");

		if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Frame: %i - Index: %i", current_frame, pathtracer.frames_accumulated);
			ImGui::Text("Delta: %.2f ms", 1000.0f * delta_time);
			ImGui::Text("Avg:   %.2f ms", 1000.0f * avg);
			ImGui::Text("Min:   %.2f ms", 1000.0f * min);
			ImGui::Text("Max:   %.2f ms", 1000.0f * max);
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

			settings_changed |= ImGui::Checkbox("Rasterize Primary Rays", &pathtracer.settings.enable_rasterization);
			settings_changed |= ImGui::Checkbox("NEE",                    &pathtracer.settings.enable_next_event_estimation);
			settings_changed |= ImGui::Checkbox("MIS",                    &pathtracer.settings.enable_multiple_importance_sampling);
			settings_changed |= ImGui::Checkbox("Update Scene",           &pathtracer.settings.enable_scene_update);
			settings_changed |= ImGui::Checkbox("SVGF",                   &pathtracer.settings.enable_svgf);
			settings_changed |= ImGui::Checkbox("Spatial Variance",       &pathtracer.settings.enable_spatial_variance);
			settings_changed |= ImGui::Checkbox("TAA",                    &pathtracer.settings.enable_taa);
			settings_changed |= ImGui::Checkbox("Demodulate Albedo",      &pathtracer.settings.demodulate_albedo);

			settings_changed |= ImGui::Combo("Reconstruction Filter", reinterpret_cast<int *>(&pathtracer.settings.reconstruction_filter), "Box\0Mitchel-Netravali\0Gaussian");

			settings_changed |= ImGui::SliderInt("A Trous iterations", &pathtracer.settings.atrous_iterations, 0, MAX_ATROUS_ITERATIONS);

			settings_changed |= ImGui::SliderFloat("Alpha colour", &pathtracer.settings.alpha_colour, 0.0f, 1.0f);
			settings_changed |= ImGui::SliderFloat("Alpha moment", &pathtracer.settings.alpha_moment, 0.0f, 1.0f);

			pathtracer.settings_changed = settings_changed;
		}

		ImGui::End();

		if (perf_test.frame_end(delta_time)) break;

		// Save Keyboard State of this frame before SDL_PumpEvents
		Input::update();

		window.gui_end();
		window.swap();
	}

	CUDAContext::destroy();

	return EXIT_SUCCESS;
}
