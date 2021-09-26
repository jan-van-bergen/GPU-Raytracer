#include <cstdio>

#include <Imgui/imgui.h>

#include "CUDA/CUDAContext.h"

#include "Pathtracer/Pathtracer.h"

#include "Config.h"

#include "Input.h"
#include "Window.h"

#include "Util/Util.h"
#include "Util/Parser.h"
#include "Util/PerfTest.h"
#include "Util/ScopeTimer.h"

extern "C" { _declspec(dllexport) unsigned NvOptimusEnablement = true; } // Forces NVIDIA driver to be used

static Window window;

static Pathtracer pathtracer;
static PerfTest   perf_test;

#define FRAMETIME_HISTORY_LENGTH 100

struct Timing {
	Uint64 start;
	Uint64 now;
	Uint64 last;

	double inv_perf_freq;
	double delta_time;

	double second;
	int frames_this_second;
	int fps;

	double avg;
	double min;
	double max;
	double history[FRAMETIME_HISTORY_LENGTH];

	int frame_index;
} static timing;

static int last_pixel_query_x;
static int last_pixel_query_y;

static void parse_args(int arg_count, char ** args);
static void capture_screen(const Window & window, const char * file_name);
static void window_resize(unsigned frame_buffer_handle, int width, int height);
static void calc_timing();
static void draw_gui();

int main(int arg_count, char ** args) {
	parse_args(arg_count, args);

	{
		ScopeTimer timer("Initialization");

		window.init("Pathtracer", config.initial_width, config.initial_height);
		window.resize_handler = &window_resize;

		CUDAContext::init();
		pathtracer.init(config.scene, config.sky, window.frame_buffer_handle, config.initial_width, config.initial_height);

		perf_test.init(&pathtracer, false, config.scene);
	}

	timing.inv_perf_freq = 1.0 / double(SDL_GetPerformanceFrequency());
	timing.start = SDL_GetPerformanceCounter();
	timing.last  = timing.start;

	// Game loop
	while (!window.is_closed) {
		perf_test.frame_begin();

		pathtracer.update((float)timing.delta_time);
		pathtracer.render();

		window.render_framebuffer();

		if (pathtracer.frames_accumulated == config.output_frame_index) {
			capture_screen(window, config.output_name);
			break; // Exit game loop and terimate
		}
		if (Input::is_key_pressed(SDL_SCANCODE_P)) {
			char screenshot_name[32] = { };
			sprintf_s(screenshot_name, "screenshot_%i.ppm", pathtracer.frames_accumulated);

			capture_screen(window, screenshot_name);
		}

		if (ImGui::IsMouseClicked(1)) {
			// Deselect current object
			pathtracer.pixel_query.pixel_index = INVALID;
			pathtracer.pixel_query.mesh_id     = INVALID;
			pathtracer.pixel_query.triangle_id = INVALID;
			pathtracer.pixel_query.material_id = INVALID;
		}

		if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse) {
			Input::mouse_position(&last_pixel_query_x, &last_pixel_query_y);

			pathtracer.set_pixel_query(last_pixel_query_x, last_pixel_query_y);
		}

		calc_timing();
		draw_gui();

		if (Input::is_key_released(SDL_SCANCODE_F5)) {
			ScopeTimer timer("Hot Reload");

			pathtracer.cuda_free();
			pathtracer.cuda_init(window.frame_buffer_handle, window.width, window.height);
		}

		if (perf_test.frame_end((float)timing.delta_time)) break;

		Input::update(); // Save Keyboard State of this frame before SDL_PumpEvents

		window.swap();
	}

	if (pathtracer.frames_accumulated < config.output_frame_index) {
		capture_screen(window, config.output_name);
	}

	CUDAContext::free();
	window.free();

	return EXIT_SUCCESS;
}

static bool atob(const char * str) {
	if (
		strcmp(str, "true") == 0 ||
		strcmp(str, "True") == 0 ||
		strcmp(str, "TRUE") == 0 ||
		strcmp(str, "1")    == 0) {
		return true;
	} else if (
		strcmp(str, "false") == 0 ||
		strcmp(str, "False") == 0 ||
		strcmp(str, "FALSE") == 0 ||
		strcmp(str, "0")     == 0) {
		return false;
	} else {
		printf("Invalid boolean argument '%s'!\n", str);
		return true;
	}
};

static void parse_args(int arg_count, char ** args) {
	struct Option {
		const char * name_short;
		const char * name_full;

		const char * help_text;

		int num_args;

		void (* action)(int arg_count, char ** args, int i);
	};

	static Array<Option> options = {
		Option { "W", "width",   "Sets the width of the window",                                          1, [](int arg_count, char ** args, int i) { config.initial_width      = atoi(args[i + 1]); } },
		Option { "H", "height",  "Sets the height of the window",                                         1, [](int arg_count, char ** args, int i) { config.initial_height     = atoi(args[i + 1]); } },
		Option { "b", "bounce",  "Sets the number of pathtracing bounces",                                1, [](int arg_count, char ** args, int i) { config.num_bounces        = Math::clamp(atoi(args[i + 1]), 0, MAX_BOUNCES - 1); } },
		Option { "N", "samples", "Sets a target number of samples to use",                                1, [](int arg_count, char ** args, int i) { config.output_frame_index = atoi(args[i + 1]); } },
		Option { "o", "output",  "Sets path to output file. Supported formats: ppm",                      1, [](int arg_count, char ** args, int i) { config.output_name        = args[i + 1]; } },
		Option { "s", "scene",   "Sets path to scene file. Supported formats: Mitsuba XML, OBJ, and PLY", 1, [](int arg_count, char ** args, int i) { config.scene              = args[i + 1]; } },
		Option { "S", "sky",     "Sets path to sky file. Supported formats: HDR",                         1, [](int arg_count, char ** args, int i) { config.sky                = args[i + 1]; } },
		Option { "b", "bvh",     "Sets type of BVH used: Supported options: bvh, sbvh, qbvh, cwbvh",      1, [](int arg_count, char ** args, int i) {
			if (strcmp(args[i + 1], "bvh") == 0) {
				config.bvh_type = BVHType::BVH;
			} else if (strcmp(args[i + 1], "sbvh") == 0) {
				config.bvh_type = BVHType::SBVH;
			} else if (strcmp(args[i + 1], "qbvh") == 0) {
				config.bvh_type = BVHType::QBVH;
			} else if (strcmp(args[i + 1], "cwbvh") == 0) {
				config.bvh_type = BVHType::CWBVH;
			} else {
				printf("'%s' is not a recognized BVH type!\n", args[i + 1]);
				abort();
			}
		} },
		Option { nullptr, "albedo", "Enables or disables albedo",                       1, [](int arg_count, char ** args, int i) { config.enable_albedo                       = atob(args[i + 1]); } },
		Option { nullptr, "nee",    "Enables or disables Next Event Estimation",        1, [](int arg_count, char ** args, int i) { config.enable_next_event_estimation        = atob(args[i + 1]); } },
		Option { nullptr, "mis",    "Enables or disables Multiple Importance Sampling", 1, [](int arg_count, char ** args, int i) { config.enable_multiple_importance_sampling = atob(args[i + 1]); } },
		Option { "O",     "optimize",    "Enables or disables BVH optimzation post-processing step",                                              1, [](int arg_count, char ** args, int i) { config.enable_bvh_optimization       = atob(args[i + 1]); } },
		Option { "Ot",    "opt-time",    "Sets time limit (in seconds) for BVH optimization",                                                     1, [](int arg_count, char ** args, int i) { config.bvh_optimizer_max_time        = atoi(args[i + 1]); } },
		Option { "Ob",    "opt-batches", "Sets a limit on the maximum number of batches used in BVH optimization",                                1, [](int arg_count, char ** args, int i) { config.bvh_optimizer_max_num_batches = atoi(args[i + 1]); } },
		Option { nullptr, "sah-node",    "Sets the SAH cost of an internal BVH node",                                                             1, [](int arg_count, char ** args, int i) { config.sah_cost_node                 = atof(args[i + 1]); } },
		Option { nullptr, "sah-leaf",    "Sets the SAH cost of a leaf BVH node",                                                                  1, [](int arg_count, char ** args, int i) { config.sah_cost_leaf                 = atof(args[i + 1]); } },
		Option { nullptr, "sbvh-alpha",  "Sets the SBVH alpha constant. An alpha of 1 results in a regular BVH, alpha of 0 results in full SBVH", 1, [](int arg_count, char ** args, int i) { config.sbvh_alpha                    = atof(args[i + 1]); } },
		Option { nullptr, "mipmap",      "Enables or disables texture mipmapping",                                                                1, [](int arg_count, char ** args, int i) { config.enable_mipmapping             = atob(args[i + 1]); } },
		Option { nullptr, "mip-filter",  "Sets the downsampling filter for creating mipmaps: Supported options: box, lanczos, kaiser",            1, [](int arg_count, char ** args, int i) {
			if (strcmp(args[i + 1], "box") == 0) {
				config.mipmap_filter = Config::MipmapFilter::BOX;
			} else if (strcmp(args[i + 1], "lanczos") == 0) {
				config.mipmap_filter = Config::MipmapFilter::LANCZOS;
			} else if (strcmp(args[i + 1], "kaiser") == 0) {
				config.mipmap_filter = Config::MipmapFilter::KAISER;
			} else {
				printf("'%s' is not a recognized Mipmap Filter!\n", args[i + 1]);
				abort();
			}
		} },
		Option { "c", "compress", "Enables or disables texture block compression", 1, [](int arg_count, char ** args, int i) { config.enable_block_compression = atob(args[i + 1]); } },
	};

	options.emplace_back("h", "help", "Displays this message", 0, [](int arg_count, char ** args, int i) {
		for (int o = 0; o < options.size(); o++) {
			const Option & option = options[o];

			if (option.name_short) {
				printf("-%s,\t--%-16s%s\n", option.name_short, option.name_full, option.help_text);
			} else {
				printf("\t--%-16s%s\n", option.name_full, option.help_text);
			}
		}
	});

	char arg_name[32] = { };

	for (int i = 1; i < arg_count; i++) {
		const char * arg     = args[i];
		int          arg_len = strlen(arg);

		sprintf_s(arg_name, "Arg %i", i);

		Parser parser = { };
		parser.init(arg, arg + arg_len, arg_name);

		if (parser.match('-')) {
			bool use_full_name = parser.match('-');

			bool match = false;

			for (int o = 0; o < options.size(); o++) {
				const Option & option = options[o];

				if (use_full_name) {
					match = strcmp(parser.cur, option.name_full) == 0;
				} else if (option.name_short) {
					match = strcmp(parser.cur, option.name_short) == 0;
				}

				if (match) {
					if (i + option.num_args >= arg_count) {
						printf("Not enough arguments provided to option '%s'!\n", option.name_full);
						return;
					}

					option.action(arg_count, args, i);
					i += option.num_args;

					break;
				}
			}

			if (!match) {
				printf("Unrecognized command line option '%s'\nUse --help for a list of valid options\n", parser.cur);
			}
		} else {
			// Without explicit option, assume scene name
			config.scene = arg;
		}
	}
}

static void capture_screen(const Window & window, const char * file_name) {
	ScopeTimer timer("Screenshot");

	int pack_alignment; glGetIntegerv(GL_PACK_ALIGNMENT, &pack_alignment);
	int window_pitch = Math::round_up(window.width * 3, pack_alignment);

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

static void window_resize(unsigned frame_buffer_handle, int width, int height) {
	pathtracer.resize_free();
	pathtracer.resize_init(frame_buffer_handle, width, height);
};

static void calc_timing() {
	// Calculate delta time
	timing.now = SDL_GetPerformanceCounter();
	timing.delta_time = double(timing.now - timing.last) * timing.inv_perf_freq;
	timing.last = timing.now;

	// Calculate average of last frames
	timing.history[timing.frame_index++ % FRAMETIME_HISTORY_LENGTH] = timing.delta_time;

	int count = timing.frame_index < FRAMETIME_HISTORY_LENGTH ? timing.frame_index : FRAMETIME_HISTORY_LENGTH;

	int min_index = INVALID;
	int max_index = INVALID;

	timing.min = INFINITY;
	timing.max = 0.0;

	for (int i = 0; i < count; i++) {
		float time = timing.history[i];
		if (time < timing.min) {
			timing.min = time;
			min_index = i;
		} else if (time > timing.max) {
			timing.max = time;
			max_index = i;
		}
	}

	timing.avg = 0.0;
	if (count <= 2) {
		for (int i = 0; i < count; i++) {
			timing.avg += timing.history[i];
		}
		timing.avg /= double(count);
	} else {
		for (int i = 0; i < count; i++) {
			if (i != min_index && i != max_index) { // For a more representative average, ignore the min and max times
				timing.avg += timing.history[i];
			}
		}
		timing.avg /= double(count - 2);
	}

	// Calculate fps
	timing.frames_this_second++;

	timing.second += timing.delta_time;
	while (timing.second >= 1.0) {
		timing.second -= 1.0;

		timing.fps = timing.frames_this_second;
		timing.frames_this_second = 0;
	}
}

static void draw_gui() {
	window.gui_begin();

	if (ImGui::Begin("Pathtracer")) {
		if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Frame: %i", pathtracer.frames_accumulated);
			ImGui::Text("Time:  %.2f s", double(timing.now - timing.start) * timing.inv_perf_freq);
			ImGui::Text("Delta: %.2f ms (%i fps)", 1000.0f * timing.delta_time, timing.fps);
			ImGui::Text("Avg:   %.2f ms", 1000.0f * timing.avg);
			ImGui::Text("Min:   %.2f ms", 1000.0f * timing.min);
			ImGui::Text("Max:   %.2f ms", 1000.0f * timing.max);
		}

		if (ImGui::CollapsingHeader("Kernels", ImGuiTreeNodeFlags_DefaultOpen)) {
			struct EventTiming {
				CUDAEvent::Desc desc;
				float           timing;
			};

			int           event_timing_count = pathtracer.event_pool.num_used - 1;
			EventTiming * event_timings = new EventTiming[event_timing_count];

			for (int i = 0; i < event_timing_count; i++) {
				event_timings[i].desc = pathtracer.event_pool.pool[i].desc;
				event_timings[i].timing = CUDAEvent::time_elapsed_between(
					pathtracer.event_pool.pool[i],
					pathtracer.event_pool.pool[i + 1]
				);
			}

			Util::stable_sort(event_timings, event_timings + event_timing_count, [](const EventTiming & a, const EventTiming & b) {
				if (a.desc.display_order == b.desc.display_order) {
					return strcmp(a.desc.category, b.desc.category) < 0;
				}
				return a.desc.display_order < b.desc.display_order;
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
						int length = strlen(event_timings[j].desc.name);
						if (length > padding) padding = length;

						time_sum += event_timings[j].timing;

						if (j < event_timing_count - 1 && strcmp(event_timings[j].desc.category, event_timings[j + 1].desc.category) != 0) break;
					}

					bool category_visible = ImGui::TreeNode(event_timings[i].desc.category, "%s: %.2f ms", event_timings[i].desc.category, time_sum);
					if (!category_visible) {
						// Skip ahead to next category
						i = j;

						continue;
					}
				}

				// Add up all timings with the same name
				float timing = 0.0f;

				while (true) {
					timing += event_timings[i].timing;

					if (i == event_timing_count - 1 || strcmp(event_timings[i].desc.name, event_timings[i+1].desc.name) != 0) break;

					i++;
				};

				ImGui::Text("%s: %*.2f ms", event_timings[i].desc.name, 5 + padding - strlen(event_timings[i].desc.name), timing);

				if (i == event_timing_count - 1) {
					ImGui::TreePop();
					break;
				}

				category_changed = strcmp(event_timings[i].desc.category, event_timings[i + 1].desc.category);
				if (category_changed) {
					ImGui::TreePop();
				}
			}

			delete [] event_timings;
		}

		if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
			switch (config.bvh_type) {
				case BVHType::BVH:	 ImGui::TextUnformatted("BVH: BVH"); break;
				case BVHType::SBVH:	 ImGui::TextUnformatted("BVH: SBVH"); break;
				case BVHType::QBVH:	 ImGui::TextUnformatted("BVH: QBVH"); break;
				case BVHType::CWBVH: ImGui::TextUnformatted("BVH: CWBVH"); break;
			}

			bool invalidated_config = ImGui::SliderInt("Num Bounces", &config.num_bounces, 0, MAX_BOUNCES);

			float fov = Math::rad_to_deg(pathtracer.scene.camera.fov);
			if (ImGui::SliderFloat("FOV", &fov, 0.0f, 179.0f)) {
				pathtracer.scene.camera.set_fov(Math::deg_to_rad(fov));
				pathtracer.invalidated_camera = true;
			}

			pathtracer.invalidated_camera |= ImGui::SliderFloat("Aperture", &pathtracer.scene.camera.aperture_radius, 0.0f, 1.0f);
			pathtracer.invalidated_camera |= ImGui::SliderFloat("Focus",    &pathtracer.scene.camera.focal_distance, 0.001f, 50.0f);

			invalidated_config |= ImGui::Checkbox("NEE", &config.enable_next_event_estimation);
			invalidated_config |= ImGui::Checkbox("MIS", &config.enable_multiple_importance_sampling);

			invalidated_config |= ImGui::Checkbox("Russian Roulete", &config.enable_russian_roulette);

			invalidated_config |= ImGui::Checkbox("Update Scene", &config.enable_scene_update);

			if (ImGui::Checkbox("SVGF", &config.enable_svgf)) {
				if (config.enable_svgf) {
					pathtracer.svgf_init();
				} else {
					pathtracer.svgf_free();
				}
				invalidated_config = true;
			}

			invalidated_config |= ImGui::Checkbox("Spatial Variance",  &config.enable_spatial_variance);
			invalidated_config |= ImGui::Checkbox("TAA",               &config.enable_taa);
			invalidated_config |= ImGui::Checkbox("Modulate Albedo",   &config.enable_albedo);

			invalidated_config |= ImGui::Combo("Reconstruction Filter", reinterpret_cast<int *>(&config.reconstruction_filter), "Box\0Tent\0Gaussian\0");

			invalidated_config |= ImGui::SliderInt("A Trous iterations", &config.num_atrous_iterations, 0, MAX_ATROUS_ITERATIONS);

			invalidated_config |= ImGui::SliderFloat("Alpha colour", &config.alpha_colour, 0.0f, 1.0f);
			invalidated_config |= ImGui::SliderFloat("Alpha moment", &config.alpha_moment, 0.0f, 1.0f);

			pathtracer.invalidated_config = invalidated_config;
		}
	}
	ImGui::End();

	if (ImGui::Begin("Scene")) {
		if (ImGui::CollapsingHeader("Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text("Has Diffuse:    %s", pathtracer.scene.has_diffuse    ? "True" : "False");
			ImGui::Text("Has Dielectric: %s", pathtracer.scene.has_dielectric ? "True" : "False");
			ImGui::Text("Has Glossy:     %s", pathtracer.scene.has_glossy     ? "True" : "False");
			ImGui::Text("Has Lights:     %s", pathtracer.scene.has_lights     ? "True" : "False");

			int triangle_count       = 0;
			int light_mesh_count     = 0;
			int light_triangle_count = 0;

			for (int i = 0; i < pathtracer.scene.meshes.size(); i++) {
				const Mesh     & mesh      = pathtracer.scene.meshes[i];
				const MeshData & mesh_data = pathtracer.scene.asset_manager.get_mesh_data(mesh.mesh_data_handle);

				triangle_count += mesh_data.triangle_count;

				if (mesh.light.weight > 0.0f) {
					light_mesh_count++;
					light_triangle_count += mesh_data.triangle_count;
				}
			}

			ImGui::Text("Meshes:          %i", int(pathtracer.scene.meshes.size()));
			ImGui::Text("Triangles:       %i", triangle_count);
			ImGui::Text("Light Meshes:    %i", light_mesh_count);
			ImGui::Text("Light Triangles: %i", light_triangle_count);
		}

		if (ImGui::CollapsingHeader("Meshes", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::BeginChild("Meshes", ImVec2(0, 200), true);

			for (int m = 0; m < pathtracer.scene.meshes.size(); m++) {
				const Mesh & mesh = pathtracer.scene.meshes[m];

				bool is_selected = pathtracer.pixel_query.mesh_id == m;

				ImGui::PushID(m);
				if (ImGui::Selectable(mesh.name ? mesh.name : "(null)", &is_selected)) {
					pathtracer.pixel_query.mesh_id     = m;
					pathtracer.pixel_query.triangle_id = INVALID;
					pathtracer.pixel_query.material_id = mesh.material_handle.handle;
				}
				ImGui::PopID();
			}

			ImGui::EndChild();
		}

		if (pathtracer.pixel_query.mesh_id != INVALID) {
			Mesh & mesh = pathtracer.scene.meshes[pathtracer.pixel_query.mesh_id];

			if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::TextUnformatted(mesh.name);

				bool mesh_changed = false;
				mesh_changed |= ImGui::DragFloat3("Position", &mesh.position.x);

				static bool dragging = false;

				if (ImGui::DragFloat3("Rotation", &mesh.euler_angles.x)) {
					mesh.euler_angles.x = Math::wrap(mesh.euler_angles.x, 0.0f, 360.0f);
					mesh.euler_angles.y = Math::wrap(mesh.euler_angles.y, 0.0f, 360.0f);
					mesh.euler_angles.z = Math::wrap(mesh.euler_angles.z, 0.0f, 360.0f);

					if (!dragging) {
						mesh.euler_angles = Quaternion::to_euler(mesh.rotation);
						mesh.euler_angles.x = Math::rad_to_deg(mesh.euler_angles.x);
						mesh.euler_angles.y = Math::rad_to_deg(mesh.euler_angles.y);
						mesh.euler_angles.z = Math::rad_to_deg(mesh.euler_angles.z);
						dragging = true;
					}

					mesh.rotation = Quaternion::from_euler(Math::deg_to_rad(mesh.euler_angles.x), Math::deg_to_rad(mesh.euler_angles.y), Math::deg_to_rad(mesh.euler_angles.z));
					mesh_changed = true;
				}

				mesh_changed |= ImGui::DragFloat("Scale", &mesh.scale, 0.1f, 0.0f, INFINITY);

				if (mesh_changed) pathtracer.invalidated_scene = true;
			}
		}

		if (pathtracer.pixel_query.material_id != INVALID) {
			Material & material = pathtracer.scene.asset_manager.get_material(MaterialHandle { pathtracer.pixel_query.material_id });

			if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::Text("Name: %s", material.name);

				bool material_changed = false;

				int material_type = int(material.type);
				if (ImGui::Combo("Type", &material_type, "Light\0Diffuse\0Dielectric\0Glossy\0")) {
					material.type = Material::Type(material_type);
					material_changed = true;
				}

				const char * texture_name = "None";
				if (material.texture_id.handle != INVALID) {
					const Texture & texture = pathtracer.scene.asset_manager.get_texture(material.texture_id);
					texture_name = texture.name;
				}

				switch (material.type) {
					case Material::Type::LIGHT: {
						material_changed |= ImGui::DragFloat3("Emission", &material.emission.x, 0.1f, 0.0f, INFINITY);
						break;
					}
					case Material::Type::DIFFUSE: {
						material_changed |= ImGui::SliderFloat3("Diffuse", &material.diffuse.x, 0.0f, 1.0f);
						material_changed |= ImGui::SliderInt   ("Texture", &material.texture_id.handle, -1, pathtracer.scene.asset_manager.textures.size() - 1, texture_name);
						break;
					}
					case Material::Type::DIELECTRIC: {
						material_changed |= ImGui::SliderFloat3("Transmittance", &material.transmittance.x,     0.0f, 1.0f);
						material_changed |= ImGui::SliderFloat ("IOR",           &material.index_of_refraction, 1.0f, 5.0f);
						break;
					}
					case Material::Type::GLOSSY: {
						material_changed |= ImGui::SliderFloat3("Diffuse",   &material.diffuse.x, 0.0f, 1.0f);
						material_changed |= ImGui::SliderInt   ("Texture",   &material.texture_id.handle, -1, pathtracer.scene.asset_manager.textures.size() - 1, texture_name);
						material_changed |= ImGui::SliderFloat3("Eta",       &material.eta.x, 1.0f, 5.0f);
						material_changed |= ImGui::SliderFloat3("K",         &material.k.x,   0.0f, 5.0f);
						material_changed |= ImGui::SliderFloat ("Roughness", &material.linear_roughness, 0.0f, 1.0f);
						break;
					}

					default: abort();
				}

				if (material_changed) pathtracer.invalidated_materials = true;
			}
		}
	}
	ImGui::End();

	if (pathtracer.pixel_query.mesh_id != INVALID) {
		Mesh & mesh = pathtracer.scene.meshes[pathtracer.pixel_query.mesh_id];
		ImDrawList * draw_list = ImGui::GetBackgroundDrawList();

		Vector4 aabb_corners[8] = {
			Vector4(mesh.aabb.min.x, mesh.aabb.min.y, mesh.aabb.min.z, 1.0f),
			Vector4(mesh.aabb.max.x, mesh.aabb.min.y, mesh.aabb.min.z, 1.0f),
			Vector4(mesh.aabb.max.x, mesh.aabb.min.y, mesh.aabb.max.z, 1.0f),
			Vector4(mesh.aabb.min.x, mesh.aabb.min.y, mesh.aabb.max.z, 1.0f),
			Vector4(mesh.aabb.min.x, mesh.aabb.max.y, mesh.aabb.min.z, 1.0f),
			Vector4(mesh.aabb.max.x, mesh.aabb.max.y, mesh.aabb.min.z, 1.0f),
			Vector4(mesh.aabb.max.x, mesh.aabb.max.y, mesh.aabb.max.z, 1.0f),
			Vector4(mesh.aabb.min.x, mesh.aabb.max.y, mesh.aabb.max.z, 1.0f)
		};

		// Transform from world space to homogeneous clip space
		for (int i = 0; i < 8; i++) {
			aabb_corners[i] = Matrix4::transform(pathtracer.scene.camera.view_projection, aabb_corners[i]);
		}

		auto draw_line_clipped = [draw_list](Vector4 a, Vector4 b, ImColor colour, float thickness = 1.0f) {
			if (a.z < pathtracer.scene.camera.near && b.z < pathtracer.scene.camera.near) return;

			// Clip against near plane only
			if (a.z < pathtracer.scene.camera.near) a = Math::lerp(a, b, Math::inv_lerp(pathtracer.scene.camera.near, a.z, b.z));
			if (b.z < pathtracer.scene.camera.near) b = Math::lerp(a, b, Math::inv_lerp(pathtracer.scene.camera.near, a.z, b.z));

			// Clip space to NDC to Window coordinates
			ImVec2 a_window = { (0.5f + 0.5f * a.x / a.w) * window.width, (0.5f - 0.5f * a.y / a.w) * window.height };
			ImVec2 b_window = { (0.5f + 0.5f * b.x / b.w) * window.width, (0.5f - 0.5f * b.y / b.w) * window.height };

			draw_list->AddLine(a_window, b_window, colour, thickness);
		};

		ImColor aabb_colour = ImColor(0.2f, 0.8f, 0.2f);

		draw_line_clipped(aabb_corners[0], aabb_corners[1], aabb_colour);
		draw_line_clipped(aabb_corners[1], aabb_corners[2], aabb_colour);
		draw_line_clipped(aabb_corners[2], aabb_corners[3], aabb_colour);
		draw_line_clipped(aabb_corners[3], aabb_corners[0], aabb_colour);
		draw_line_clipped(aabb_corners[4], aabb_corners[5], aabb_colour);
		draw_line_clipped(aabb_corners[5], aabb_corners[6], aabb_colour);
		draw_line_clipped(aabb_corners[6], aabb_corners[7], aabb_colour);
		draw_line_clipped(aabb_corners[7], aabb_corners[4], aabb_colour);
		draw_line_clipped(aabb_corners[0], aabb_corners[4], aabb_colour);
		draw_line_clipped(aabb_corners[1], aabb_corners[5], aabb_colour);
		draw_line_clipped(aabb_corners[2], aabb_corners[6], aabb_colour);
		draw_line_clipped(aabb_corners[3], aabb_corners[7], aabb_colour);

		if (pathtracer.pixel_query.triangle_id != INVALID) {
			const MeshData & mesh_data = pathtracer.scene.asset_manager.get_mesh_data(mesh.mesh_data_handle);

			int              index    = mesh_data.bvh.indices[pathtracer.pixel_query.triangle_id - pathtracer.mesh_data_triangle_offsets[mesh.mesh_data_handle.handle]];
			const Triangle & triangle = mesh_data.triangles[index];

			int mouse_x, mouse_y;
			Input::mouse_position(&mouse_x, &mouse_y);

			if (Vector2::length(Vector2(mouse_x, mouse_y) - Vector2(last_pixel_query_x, last_pixel_query_y)) < 50.0f) {
				ImGui::BeginTooltip();
				ImGui::Text("Distance: %f", Vector3::length(triangle.get_center() - pathtracer.scene.camera.position));
				ImGui::EndTooltip();
			}

			Vector4 triangle_positions[3] = {
				Vector4(triangle.position_0.x, triangle.position_0.y, triangle.position_0.z, 1.0f),
				Vector4(triangle.position_1.x, triangle.position_1.y, triangle.position_1.z, 1.0f),
				Vector4(triangle.position_2.x, triangle.position_2.y, triangle.position_2.z, 1.0f)
			};

			Vector4 triangle_normals[3] = {
				Vector4(triangle.normal_0.x, triangle.normal_0.y, triangle.normal_0.z, 0.0f),
				Vector4(triangle.normal_1.x, triangle.normal_1.y, triangle.normal_1.z, 0.0f),
				Vector4(triangle.normal_2.x, triangle.normal_2.y, triangle.normal_2.z, 0.0f)
			};

			for (int i = 0; i < 3; i++) {
				triangle_positions[i] = Matrix4::transform(pathtracer.scene.camera.view_projection * mesh.transform, triangle_positions[i]);
				triangle_normals  [i] = Matrix4::transform(pathtracer.scene.camera.view_projection * mesh.transform, triangle_normals  [i]);
			}

			ImColor triangle_colour = ImColor(0.8f, 0.2f, 0.8f);

			draw_line_clipped(triangle_positions[0], triangle_positions[1], triangle_colour);
			draw_line_clipped(triangle_positions[1], triangle_positions[2], triangle_colour);
			draw_line_clipped(triangle_positions[2], triangle_positions[0], triangle_colour);

			ImColor normal_colour = ImColor(0.2f, 0.5f, 0.8f);

			draw_line_clipped(triangle_positions[0], triangle_positions[0] + 0.1f * triangle_normals[0], normal_colour);
			draw_line_clipped(triangle_positions[1], triangle_positions[1] + 0.1f * triangle_normals[1], normal_colour);
			draw_line_clipped(triangle_positions[2], triangle_positions[2] + 0.1f * triangle_normals[2], normal_colour);
		}
	}

	window.gui_end();
}
