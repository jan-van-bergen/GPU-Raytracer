#include <cstdio>

#include <Imgui/imgui.h>

#include "CUDA/CUDAContext.h"

#include "Pathtracer/Pathtracer.h"

#include "Input.h"
#include "Window.h"

#include "Util/Util.h"
#include "Util/Random.h"
#include "Util/PerfTest.h"
#include "Util/ScopeTimer.h"

// Forces NVIDIA driver to be used 
extern "C" { _declspec(dllexport) unsigned NvOptimusEnablement = true; }

static void capture_screen(const Window & window, const char * file_name) {
	ScopeTimer timer("Screenshot");
	
	int pack_alignment; glGetIntegerv(GL_PACK_ALIGNMENT, &pack_alignment);
	int window_pitch = Math::divide_round_up(window.width * 3, pack_alignment) * pack_alignment;

	unsigned char * data = new unsigned char[window_pitch * window.height];
	unsigned char * temp = new unsigned char[window_pitch];
			
	window.read_frame_buffer(data);
	
	// Flip image vertically
	for (int j = 0; j < window.height / 2; j++) {
		unsigned char * row_top    = data +                  j      * window_pitch;
		unsigned char * row_bottom = data + (window.height - j - 1) * window_pitch;

		memcpy(temp,       row_top,    window_pitch);
		memcpy(row_top,    row_bottom, window_pitch);
		memcpy(row_bottom, temp,       window_pitch);
	}

	// Remove pack alignment
	for (int j = 1; j < window.height; j++) {
		memmove(data + j * window.width * 3, data + j * window_pitch, window.width * 3);
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
		DATA_PATH("Diamond.obj"),
		DATA_PATH("Lantern.obj")
	};
	const char * sky_filename = DATA_PATH("Sky_Probes/sky_15.hdr");

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

			struct EventTiming {
				CUDAEvent::Info info;
				float           time;
			};
			
			int           event_timing_count = CUDAEvent::event_pool_num_used - 1;
			EventTiming * event_timings = new EventTiming[event_timing_count];

			for (int i = 0; i < event_timing_count; i++) {
				event_timings[i].info = CUDAEvent::event_pool[i].info;
				event_timings[i].time = CUDAEvent::time_elapsed_between(CUDAEvent::event_pool[i], CUDAEvent::event_pool[i + 1]);
			}

			std::sort(event_timings, event_timings + event_timing_count, [](const EventTiming & a, const EventTiming & b) {
				if (a.info.display_order == b.info.display_order) {
					return strcmp(a.info.category, b.info.category) < 0;
				}
				return a.info.display_order < b.info.display_order;
			});

			bool category_changed = true;
			int  padding;

			// Display Profile timings per category
			for (int i = 0; i < event_timing_count; i++) {
				if (category_changed) {
					padding = 0;

					// Sum the times of all events in the new Category so it can be displayed in the header
					float time_sum = 0.0f;

					int j;
					for (j = i; j < event_timing_count; j++) {
						int length = strlen(event_timings[j].info.name);
						if (length > padding) padding = length;

						time_sum += event_timings[j].time;

						if (j < event_timing_count - 1 && strcmp(event_timings[j].info.category, event_timings[j + 1].info.category) != 0) break;
					}

					bool category_visible = ImGui::TreeNode(event_timings[i].info.category, "%s: %.2f ms", event_timings[i].info.category, time_sum);
					if (!category_visible) {
						// Skip ahead to next category
						i = j;

						continue;
					}
				}

				// Add up all timings with the same name
				float time = 0.0f;

				while (true) {
					time += event_timings[i].time;

					if (i == event_timing_count - 1 || strcmp(event_timings[i].info.name, event_timings[i+1].info.name) != 0) break;

					i++;
				};

				ImGui::Text("%s: %*.2f ms", event_timings[i].info.name, 5 + padding - strlen(event_timings[i].info.name), time);

				if (i == event_timing_count - 1) {
					ImGui::TreePop();
					break;
				}

				category_changed = strcmp(event_timings[i].info.category, event_timings[i + 1].info.category);
				if (category_changed) {
					ImGui::TreePop();
				}
			}

			ImGui::EndChild();

			delete [] event_timings;
		}

		if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
			bool settings_changed = false;
			
			settings_changed |= ImGui::SliderInt("Num Bounces", &pathtracer.settings.num_bounces, 0, MAX_BOUNCES);

			settings_changed |= ImGui::SliderFloat("Aperture", &pathtracer.settings.camera_aperture,       0.0f,    1.0f);
			settings_changed |= ImGui::SliderFloat("Focus",    &pathtracer.settings.camera_focal_distance, 0.001f, 50.0f);

			settings_changed |= ImGui::Checkbox("Rasterize Primary Rays", &pathtracer.settings.enable_rasterization);
			settings_changed |= ImGui::Checkbox("NEE",                    &pathtracer.settings.enable_next_event_estimation);
			settings_changed |= ImGui::Checkbox("MIS",                    &pathtracer.settings.enable_multiple_importance_sampling);
			settings_changed |= ImGui::Checkbox("Update Scene",           &pathtracer.settings.enable_scene_update);
			settings_changed |= ImGui::Checkbox("SVGF",                   &pathtracer.settings.enable_svgf);
			settings_changed |= ImGui::Checkbox("Spatial Variance",       &pathtracer.settings.enable_spatial_variance);
			settings_changed |= ImGui::Checkbox("TAA",                    &pathtracer.settings.enable_taa);
			settings_changed |= ImGui::Checkbox("Demodulate Albedo",      &pathtracer.settings.demodulate_albedo);

			settings_changed |= ImGui::Combo("Reconstruction Filter", reinterpret_cast<int *>(&pathtracer.settings.reconstruction_filter), "Box\0Gaussian\0");

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
