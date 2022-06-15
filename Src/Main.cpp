#include <cstdio>

#include <Imgui/imgui.h>

#include "Renderer/Integrators/AO.h"
#include "Renderer/Integrators/Pathtracer.h"

#include "Config.h"
#include "Args.h"

#include "Core/Sort.h"
#include "Core/Parser.h"
#include "Core/Timer.h"
#include "Core/Allocators/StackAllocator.h"

#include "Input.h"
#include "Window.h"

#include "Exporters/EXRExporter.h"
#include "Exporters/PPMExporter.h"

#include "Util/Util.h"
#include "Util/PerfTest.h"

extern "C" { _declspec(dllexport) unsigned NvOptimusEnablement = true; } // Forces NVIDIA driver to be used

static constexpr int FRAMETIME_HISTORY_LENGTH = 100;

static constexpr float SCREENSHOT_FADE_TIME = 5.0f;

struct Timing {
	Uint64 start;
	Uint64 now;
	Uint64 last;

	Uint64 time_of_last_screenshot;

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

static bool integrator_change_requested = false;

static void capture_screen(const Window & window, const Integrator & integrator, const String & filename);
static void calc_timing();
static void draw_gui(Window & window, Integrator & integrator);

static void init_integrator(OwnPtr<Integrator> & integrator, const Window & window, Scene & scene) {
	if (integrator) {
		integrator->cuda_free();
	}

	switch (cpu_config.integrator) {
		case IntegratorType::PATHTRACER: integrator = make_owned<Pathtracer>(window.frame_buffer_handle, window.width, window.height, scene); break;
		case IntegratorType::AO:         integrator = make_owned<AO>        (window.frame_buffer_handle, window.width, window.height, scene); break;
		default: ASSERT_UNREACHABLE();
	}
}

int main(int num_args, char ** args) {
	Args::parse(num_args, args);
	if (cpu_config.scene_filenames.size() == 0) {
		cpu_config.scene_filenames.push_back("Data/sponza/scene.xml"_sv);
	}
	if (cpu_config.sky_filename.is_empty()) {
		cpu_config.sky_filename = "Data/Skies/sky_15.hdr"_sv;
	}

	Timer timer = { };
	timer.start();

	ThreadPool::init();

	for (int i = 1; i < PMJ_NUM_SEQUENCES; i++) {
		ThreadPool::submit([i]() {
			PMJ::shuffle(i);
		});
	}

	Window window("Pathtracer"_sv, cpu_config.initial_width, cpu_config.initial_height);

	CUDAContext::init();

	LinearAllocator<MEGABYTES(1)> scene_allocator;
	Scene scene(&scene_allocator);

	OwnPtr<Integrator> integrator = nullptr;

	window.resize_handler = [&integrator](unsigned frame_buffer_handle, int width, int height) {
		if (integrator) {
			integrator->resize_free();
			integrator->resize_init(frame_buffer_handle, width, height);
		}
	};
	window.set_size(cpu_config.initial_width, cpu_config.initial_height);
	window.show();

	init_integrator(integrator, window, scene);

	PerfTest perf_test(*integrator.get(), false, cpu_config.scene_filenames[0].view());

	ThreadPool::free();

	size_t initialization_time = timer.stop();
	Timer::print_named_duration("Initialization"_sv, initialization_time);

	timing.inv_perf_freq = 1.0 / double(SDL_GetPerformanceFrequency());
	timing.start = SDL_GetPerformanceCounter();
	timing.last  = timing.start;
	timing.time_of_last_screenshot = INVALID;

	LinearAllocator<MEGABYTES(16)> frame_allocator;

	// Render loop
	while (!window.is_closed) {
		perf_test.frame_begin();

		if (integrator_change_requested) {
			integrator_change_requested = false;
			init_integrator(integrator, window, scene);
		}

		integrator->update((float)timing.delta_time, &frame_allocator);
		integrator->render();

		window.render_framebuffer();

		if (integrator->sample_index == cpu_config.output_sample_index) {
			capture_screen(window, *integrator.get(), cpu_config.output_filename);
			break; // Exit render loop and terimate
		}
		if (Input::is_key_pressed(SDL_SCANCODE_P)) {
			StringView ext = { };
			switch (cpu_config.screenshot_format) {
				case OutputFormat::EXR: ext = "exr"_sv; break;
				case OutputFormat::PPM: ext = "ppm"_sv; break;
				default: ASSERT_UNREACHABLE();
			}

			StackAllocator<BYTES(128)> allocator;
			String screenshot_name = Format(&allocator).format("screenshot_{}.{}"_sv, integrator->sample_index, ext);
			capture_screen(window, *integrator.get(), screenshot_name);

			timing.time_of_last_screenshot = timing.now;
		}

		if (ImGui::IsMouseClicked(1)) {
			// Deselect current object
			integrator->pixel_query.pixel_index = INVALID;
			integrator->pixel_query.mesh_id     = INVALID;
			integrator->pixel_query.triangle_id = INVALID;
		}

		if (ImGui::IsMouseClicked(0) && !ImGui::GetIO().WantCaptureMouse) {
			Input::mouse_position(&last_pixel_query_x, &last_pixel_query_y);

			integrator->set_pixel_query(last_pixel_query_x, last_pixel_query_y);
		}

		calc_timing();
		draw_gui(window, *integrator.get());

		if (Input::is_key_released(SDL_SCANCODE_F5)) {
			ScopeTimer timer("Hot Reload"_sv);

			integrator->cuda_free();
			integrator->cuda_init(window.frame_buffer_handle, window.width, window.height);
		}

		if (perf_test.frame_end((float)timing.delta_time)) break;

		Input::update(); // Save Keyboard State of this frame before SDL_PumpEvents

		window.swap();

		frame_allocator.reset();
	}

	// Free Integrator before freeing CUDA Context
	integrator = nullptr;

	CUDAContext::free();

	return EXIT_SUCCESS;
}

static void capture_screen(const Window & window, const Integrator & integrator, const String & filename) {
	ScopeTimer timer("Screenshot"_sv);

	using Exporter = void (*)(const String & filename, int pitch, int width, int height, const Array<Vector3> & data);
	Exporter exporter = nullptr;

	bool hdr = false;

	StringView file_extension = Util::get_file_extension(filename.view());
	if (file_extension == "ppm"_sv) {
		exporter = PPMExporter::save;
		hdr      = false;
	} else if (file_extension == "exr"_sv) {
		exporter = EXRExporter::save;
		hdr      = true;
	} else {
		IO::print("WARNING: Unsupported output file extension: {}!\n"_sv, file_extension);
		return;
	}

	int pitch = 0;
	Array<Vector3> data = window.read_frame_buffer(hdr, pitch);

	exporter(filename, pitch, window.width, window.height, data);

	auto export_aov = [&integrator](AOVType aov_type, const String & filename) {
		if (!integrator.aov_is_enabled(aov_type)) return;

		const AOV & aov = integrator.get_aov(aov_type);

		Array<float4> aov_raw(integrator.screen_pitch * integrator.screen_height);
		CUDAMemory::memcpy(aov_raw.data(), aov.accumulator, integrator.screen_pitch * integrator.screen_height);

		Array<Vector3> aov_data(integrator.screen_width * integrator.screen_height);
		for (int y = 0; y < integrator.screen_height; y++) {
			for (int x = 0; x < integrator.screen_width; x++) {
				aov_data[x + y * integrator.screen_width] = Vector3(
					aov_raw[x + y * integrator.screen_pitch].x,
					aov_raw[x + y * integrator.screen_pitch].y,
					aov_raw[x + y * integrator.screen_pitch].z
				);
			}
		}
		EXRExporter::save(filename, integrator.screen_width, integrator.screen_width, integrator.screen_height, aov_data);
	};

	export_aov(AOVType::ALBEDO,   "albedo.exr"_sv);
	export_aov(AOVType::NORMAL,   "normal.exr"_sv);
	export_aov(AOVType::POSITION, "position.exr"_sv);
}

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
		double time = timing.history[i];
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

// Helper function to convert any enum to int representation, show in ImGui ComboBox, and convert back to enum representation
template<typename Enum>
static bool ImGui_Combo(const char * label, Enum * current_item, const char * items_separated_by_zeros, int height_in_items = -1) {
	int as_int = int(*current_item);
	if (ImGui::Combo(label, &as_int, items_separated_by_zeros, height_in_items)) {
		*current_item = Enum(as_int);
		return true;
	} else {
		return false;
	}
}

// Helper function for displaying a combobox for selecting Materials, Textures, and Media
template<typename T, typename Callback>
static int ImGui_Combo(const char * label, const char * preview, const Array<T> & data, bool allow_none, int selected_index, Callback on_select) {
	if (ImGui::BeginCombo(label, preview)) {
		if (allow_none) {
			bool is_selected = selected_index == INVALID;
			if (ImGui::Selectable("None", &is_selected)) {
				selected_index = INVALID;
				on_select(selected_index);
			}
		}
		for (int i = 0; i < data.size(); i++) {
			bool is_selected = i == selected_index;
			ImGui::PushID(i);
			if (ImGui::Selectable(data[i].name.c_str(), &is_selected)) {
				selected_index = i;
				on_select(selected_index);
			}
			ImGui::PopID();
		}
		ImGui::EndCombo();
	}
	return selected_index;
}

static void draw_gui(Window & window, Integrator & integrator) {
	window.gui_begin();

	if (ImGui::Begin("Config")) {
		if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
			size_t time_in_seconds = size_t(double(timing.now - timing.start) * timing.inv_perf_freq);
			size_t time_in_minutes = time_in_seconds / 60;
			size_t time_in_hours   = time_in_minutes / 60;

			ImGui::Text("Frame: %i", integrator.sample_index);
			ImGui::Text("Time:  %0.2llu:%0.2llu:%0.2llu", time_in_hours, time_in_minutes % 60, time_in_seconds % 60);
			ImGui::Text("Delta: %.2f ms (%i fps)", 1000.0f * timing.delta_time, timing.fps);
			ImGui::Text("Avg:   %.2f ms", 1000.0f * timing.avg);
			ImGui::Text("Min:   %.2f ms", 1000.0f * timing.min);
			ImGui::Text("Max:   %.2f ms", 1000.0f * timing.max);

			switch (cpu_config.bvh_type) {
				case BVHType::BVH:  ImGui::TextUnformatted("BVH:   BVH");  break;
				case BVHType::SBVH: ImGui::TextUnformatted("BVH:   SBVH"); break;
				case BVHType::BVH4: ImGui::TextUnformatted("BVH:   BVH4"); break;
				case BVHType::BVH8: ImGui::TextUnformatted("BVH:   BVH8"); break;
			}
		}

		if (ImGui::CollapsingHeader("Kernel Timings") && integrator.event_pool.num_used > 0) {
			struct EventTiming {
				const CUDAEvent::Desc * desc;
				float                   timing;
			};

			Array<EventTiming> event_timings(integrator.event_pool.num_used - 1);

			for (size_t i = 0; i < event_timings.size(); i++) {
				event_timings[i].desc = integrator.event_pool.pool[i].desc;
				event_timings[i].timing = CUDAEvent::time_elapsed_between(
					integrator.event_pool.pool[i],
					integrator.event_pool.pool[i + 1]
				);
			}

			Sort::stable_sort(event_timings.begin(), event_timings.end(), [](const EventTiming & a, const EventTiming & b) {
				if (a.desc->display_order == b.desc->display_order) {
					return a.desc->category < b.desc->category;
				}
				return a.desc->display_order < b.desc->display_order;
			});

			bool category_changed = true;
			int  padding = 0;

			// Display Profile timings per category
			for (size_t i = 0; i < event_timings.size(); i++) {
				if (category_changed) {
					padding = 0;

					// Sum the times of all events in the new Category so it can be displayed in the header
					float time_sum = 0.0f;

					size_t j;
					for (j = i; j < event_timings.size(); j++) {
						int length = int(event_timings[j].desc->name.size());
						if (length > padding) padding = length;

						time_sum += event_timings[j].timing;

						if (j < event_timings.size() - 1 && event_timings[j].desc->category != event_timings[j + 1].desc->category) break;
					}

					bool category_visible = ImGui::TreeNode(event_timings[i].desc->category.data(), "%s: %.2f ms", event_timings[i].desc->category.data(), time_sum);
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

					if (i == event_timings.size() - 1 || event_timings[i].desc->name != event_timings[i+1].desc->name) break;

					i++;
				};

				ImGui::Text("%s: %*.2f ms", event_timings[i].desc->name.data(), 5 + padding - int(event_timings[i].desc->name.size()), timing);

				if (i == event_timings.size() - 1) {
					ImGui::TreePop();
					break;
				}

				category_changed = event_timings[i].desc->category != event_timings[i + 1].desc->category;
				if (category_changed) {
					ImGui::TreePop();
				}
			}
		}

		if (ImGui::CollapsingHeader("Renderer", ImGuiTreeNodeFlags_DefaultOpen)) {
			IntegratorType previous_integrator = cpu_config.integrator;
			if (ImGui_Combo("Integrator", &cpu_config.integrator, "Pathtracer\0AO\0")) {
				if (cpu_config.integrator != previous_integrator) {
					integrator_change_requested = true;
					ImGui::TextUnformatted("Loading Integrator...");
				}
			}

			integrator.invalidated_gpu_config |= ImGui_Combo("Reconstruction Filter", &gpu_config.reconstruction_filter, "Box\0Tent\0Gaussian\0");

			ImGui_Combo("Output Format", &cpu_config.screenshot_format, "EXR\0PPM\0");
		}

		integrator.render_gui();
	}
	ImGui::End();

	if (ImGui::Begin("Scene")) {
		if (ImGui::CollapsingHeader("General")) {
			ImGui::Text("Has Diffuse:     %s", integrator.scene.has_diffuse    ? "True" : "False");
			ImGui::Text("Has Plastic:     %s", integrator.scene.has_plastic    ? "True" : "False");
			ImGui::Text("Has Dielectric:  %s", integrator.scene.has_dielectric ? "True" : "False");
			ImGui::Text("Has Conductor:   %s", integrator.scene.has_conductor  ? "True" : "False");
			ImGui::Text("Has Lights:      %s", integrator.scene.has_lights     ? "True" : "False");

			size_t triangle_count       = 0;
			size_t light_mesh_count     = 0;
			size_t light_triangle_count = 0;

			for (int i = 0; i < integrator.scene.meshes.size(); i++) {
				const Mesh     & mesh      = integrator.scene.meshes[i];
				const MeshData & mesh_data = integrator.scene.asset_manager.get_mesh_data(mesh.mesh_data_handle);

				triangle_count += mesh_data.triangles.size();

				if (mesh.light.weight > 0.0f) {
					light_mesh_count++;
					light_triangle_count += mesh_data.triangles.size();
				}
			}

			ImGui::Text("Meshes:          %zu", integrator.scene.meshes.size());
			ImGui::Text("Triangles:       %zu", triangle_count);
			ImGui::Text("Light Meshes:    %zu", light_mesh_count);
			ImGui::Text("Light Triangles: %zu", light_triangle_count);
			ImGui::Separator();

			ImGui::Checkbox("Update Scene", &cpu_config.enable_scene_update);

			integrator.invalidated_sky = ImGui::DragFloat("Sky Scale", &integrator.scene.sky.scale, 0.01f, 0.0f, INFINITY);
		}

		if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
			float fov = Math::rad_to_deg(integrator.scene.camera.fov);
			if (ImGui::SliderFloat("FOV", &fov, 0.0f, 179.0f)) {
				integrator.scene.camera.set_fov(Math::deg_to_rad(fov));
				integrator.invalidated_camera = true;
			}

			integrator.invalidated_camera |= ImGui::SliderFloat("Aperture", &integrator.scene.camera.aperture_radius, 0.0f, 1.0f);
			integrator.invalidated_camera |= ImGui::SliderFloat("Focus",    &integrator.scene.camera.focal_distance, 0.001f, 50.0f);
		}

		if (ImGui::CollapsingHeader("Meshes", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::BeginChild("Meshes", ImVec2(0, 200), true);

			for (int m = 0; m < integrator.scene.meshes.size(); m++) {
				const Mesh & mesh = integrator.scene.meshes[m];

				bool is_selected = integrator.pixel_query.mesh_id == m;

				ImGui::PushID(m);
				if (ImGui::Selectable(mesh.name.data(), &is_selected)) {
					integrator.pixel_query.mesh_id     = m;
					integrator.pixel_query.triangle_id = INVALID;
				}
				ImGui::PopID();
			}

			ImGui::EndChild();
		}

		if (integrator.pixel_query.mesh_id != INVALID) {
			Mesh & mesh = integrator.scene.meshes[integrator.pixel_query.mesh_id];

			if (ImGui::CollapsingHeader("Mesh", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::TextUnformatted(mesh.name.data());

				bool mesh_changed = false;
				mesh_changed |= ImGui::DragFloat3("Position", &mesh.position.x);

				if (ImGui::DragFloat3("Rotation", &mesh.euler_angles.x)) {
					mesh.euler_angles.x = Math::wrap(mesh.euler_angles.x, 0.0f, 360.0f);
					mesh.euler_angles.y = Math::wrap(mesh.euler_angles.y, 0.0f, 360.0f);
					mesh.euler_angles.z = Math::wrap(mesh.euler_angles.z, 0.0f, 360.0f);

					mesh.rotation = Quaternion::from_euler(
						Math::deg_to_rad(mesh.euler_angles.x),
						Math::deg_to_rad(mesh.euler_angles.y),
						Math::deg_to_rad(mesh.euler_angles.z)
					);
					mesh_changed = true;
				}

				mesh_changed |= ImGui::DragFloat("Scale", &mesh.scale, 0.1f, 0.0f, INFINITY);

				if (mesh_changed) integrator.invalidated_scene = true;

				ImGui::Separator();

				if (ImGui::Button("+##NewMaterial")) {
					Material new_material = { };
					new_material.name = Format().format("Material {}"_sv, integrator.scene.asset_manager.materials.size());
					mesh.material_handle = integrator.scene.asset_manager.add_material(std::move(new_material));

					integrator.free_materials();
					integrator.init_materials();
					integrator.invalidated_materials = true;
				}
				ImGui::SameLine();

				const char * material_name = integrator.scene.asset_manager.get_material(mesh.material_handle).name.c_str();
				mesh.material_handle.handle = ImGui_Combo("Material", material_name, integrator.scene.asset_manager.materials, false, mesh.material_handle.handle, [&integrator](int index) {
					integrator.invalidated_scene     = true;
					integrator.invalidated_materials = true;
				});
			}

			if (mesh.material_handle.handle != INVALID) {
				Material & material = integrator.scene.asset_manager.get_material(mesh.material_handle);

				if (ImGui::CollapsingHeader("Material##MaterialHeader", ImGuiTreeNodeFlags_DefaultOpen)) {
					ImGui::Text("Name: %s", material.name.data());

					integrator.invalidated_materials |= ImGui_Combo("Type", &material.type, "Light\0Diffuse\0Plastic\0Dielectric\0Conductor\0");

					const char * texture_name = "None";
					if (material.texture_handle.handle != INVALID) {
						texture_name = integrator.scene.asset_manager.get_texture(material.texture_handle).name.c_str();
					}

					const char * medium_name = "None";
					if (material.medium_handle.handle != INVALID) {
						medium_name = integrator.scene.asset_manager.get_medium(material.medium_handle).name.c_str();
					}

					switch (material.type) {
						case Material::Type::LIGHT: {
							integrator.invalidated_materials |= ImGui::DragFloat3("Emission", &material.emission.x, 0.1f, 0.0f, INFINITY);
							break;
						}
						case Material::Type::DIFFUSE: {
							integrator.invalidated_materials |= ImGui::ColorEdit3("Diffuse", &material.diffuse.x);
							material.texture_handle.handle = ImGui_Combo("Texture", texture_name, integrator.scene.asset_manager.textures, true, material.texture_handle.handle, [&integrator](int index) {
								integrator.invalidated_materials = true;
							});
							break;
						}
						case Material::Type::PLASTIC: {
							integrator.invalidated_materials |= ImGui::ColorEdit3 ("Diffuse", &material.diffuse.x);
							material.texture_handle.handle = ImGui_Combo("Texture", texture_name, integrator.scene.asset_manager.textures, true, material.texture_handle.handle, [&integrator](int index) {
								integrator.invalidated_materials = true;
							});
							integrator.invalidated_materials |= ImGui::SliderFloat("Roughness", &material.linear_roughness, 0.0f, 1.0f);
							break;
						}
						case Material::Type::DIELECTRIC: {
							if (ImGui::Button("+##NewMedium")) {
								Medium new_medium = { };
								new_medium.name = Format().format("Material {}"_sv, integrator.scene.asset_manager.media.size());
								material.medium_handle = integrator.scene.asset_manager.add_medium(std::move(new_medium));

								integrator.free_materials();
								integrator.init_materials();
								integrator.invalidated_mediums   = true;
								integrator.invalidated_materials = true;
							}
							ImGui::SameLine();

							material.medium_handle.handle = ImGui_Combo("Medium", medium_name, integrator.scene.asset_manager.media, true, material.medium_handle.handle, [&integrator](int index) {
								integrator.invalidated_materials = true;
							});
							integrator.invalidated_materials |= ImGui::SliderFloat("IOR",       &material.index_of_refraction, 1.0f, 2.5f);
							integrator.invalidated_materials |= ImGui::SliderFloat("Roughness", &material.linear_roughness,    0.0f, 1.0f);
							break;
						}
						case Material::Type::CONDUCTOR: {
							integrator.invalidated_materials |= ImGui::SliderFloat3("Eta",       &material.eta.x, 0.0f, 4.0f);
							integrator.invalidated_materials |= ImGui::SliderFloat3("K",         &material.k.x,   0.0f, 8.0f);
							integrator.invalidated_materials |= ImGui::SliderFloat ("Roughness", &material.linear_roughness, 0.0f, 1.0f);
							break;
						}
						default: ASSERT_UNREACHABLE();
					}

					if (material.medium_handle.handle != INVALID && ImGui::CollapsingHeader("Medium##MediumHeader", ImGuiTreeNodeFlags_DefaultOpen)) {
						Medium & medium = integrator.scene.asset_manager.get_medium(material.medium_handle);

						Vector3 sigma_a = { };
						Vector3 sigma_s = { };
						medium.to_sigmas(sigma_a, sigma_s);

						Vector3 sigma_t = sigma_a + sigma_s;
						ImGui::Text("Sigma A: %.3f, %.3f, %.3f", sigma_a.x, sigma_a.y, sigma_a.z);
						ImGui::Text("Sigma S: %.3f, %.3f, %.3f", sigma_s.x, sigma_s.y, sigma_s.z);
						ImGui::Text("Sigma T: %.3f, %.3f, %.3f", sigma_t.x, sigma_t.y, sigma_t.z);

						integrator.invalidated_mediums |= ImGui::ColorEdit3 ("Albedo",   &medium.C.x);
						integrator.invalidated_mediums |= ImGui::DragFloat3 ("MFP",      &medium.mfp.x, 0.01f, 0.0f, INFINITY);
						integrator.invalidated_mediums |= ImGui::SliderFloat("Phase g",  &medium.g,  -1.0f,  1.0f);
					}
				}
			}
		}
	}
	ImGui::End();

	if (integrator.pixel_query.mesh_id != INVALID) {
		Mesh & mesh = integrator.scene.meshes[integrator.pixel_query.mesh_id];
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
			aabb_corners[i] = Matrix4::transform(integrator.scene.camera.view_projection, aabb_corners[i]);
		}

		auto draw_line_clipped = [draw_list, &window, near_plane = integrator.scene.camera.near_plane](Vector4 a, Vector4 b, ImColor colour, float thickness = 1.0f) {
			if (a.z < near_plane && b.z < near_plane) return;

			// Clip against near plane only
			if (a.z < near_plane) a = Math::lerp(a, b, Math::inv_lerp(near_plane, a.z, b.z));
			if (b.z < near_plane) b = Math::lerp(a, b, Math::inv_lerp(near_plane, a.z, b.z));

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

		if (integrator.pixel_query.triangle_id != INVALID) {
			const MeshData & mesh_data = integrator.scene.asset_manager.get_mesh_data(mesh.mesh_data_handle);

			int              index    = mesh_data.bvh->indices[integrator.pixel_query.triangle_id - integrator.mesh_data_triangle_offsets[mesh.mesh_data_handle.handle]];
			const Triangle & triangle = mesh_data.triangles[index];

			int mouse_x, mouse_y;
			Input::mouse_position(&mouse_x, &mouse_y);

			if (Vector2::length(Vector2(float(mouse_x), float(mouse_y)) - Vector2(float(last_pixel_query_x), float(last_pixel_query_y))) < 50.0f) {
				Vector3 triangle_center_world = Matrix4::transform_position(mesh.transform, triangle.get_center());

				ImGui::BeginTooltip();
				ImGui::Text("Distance: %f", Vector3::length(triangle_center_world - integrator.scene.camera.position));
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
				triangle_positions[i] = Matrix4::transform(integrator.scene.camera.view_projection * mesh.transform, triangle_positions[i]);
				triangle_normals  [i] = Matrix4::transform(integrator.scene.camera.view_projection * mesh.transform, triangle_normals  [i]);
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

	double time_since_last_screenshot = (timing.now - timing.time_of_last_screenshot) * timing.inv_perf_freq;
	if (time_since_last_screenshot >= 0.0f && time_since_last_screenshot < SCREENSHOT_FADE_TIME) {
		unsigned colour = Math::lerp(0xff, 0, time_since_last_screenshot / SCREENSHOT_FADE_TIME);
		colour |= colour << 8;
		colour |= colour << 16;

		const char * text = "Screenshot taken";
		float text_width = ImGui::CalcTextSize(text).x;

		ImDrawList * draw_list = ImGui::GetForegroundDrawList();
		draw_list->AddText(ImVec2(0.5f * (window.width - text_width), 0.0f), colour, text);
	}

	window.gui_end();
}
