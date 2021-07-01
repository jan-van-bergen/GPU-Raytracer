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

	const char * scene_filename = "C:/Dev/Git/Advanced_Graphics/Models/staircase2/scene.xml";
	const char * sky_filename = DATA_PATH("Sky_Probes/sky_15.hdr");
	
	{
		ScopeTimer timer("Initialization");
	
		CUDAContext::init();
		pathtracer.init(scene_filename, sky_filename, window.frame_buffer_handle);
	
		perf_test.init(&pathtracer, false, scene_filename);	
		Random::init(1337);
	}

	window.resize_handler = &window_resize;
	
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

		if (ImGui::Begin("Pathtracer")) {
			if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) {
				ImGui::Text("Frame: %i - Index: %i", current_frame, pathtracer.frames_accumulated);
				ImGui::Text("Delta: %.2f ms", 1000.0f * delta_time);
				ImGui::Text("Avg:   %.2f ms", 1000.0f * avg);
				ImGui::Text("Min:   %.2f ms", 1000.0f * min);
				ImGui::Text("Max:   %.2f ms", 1000.0f * max);
				ImGui::Text("FPS: %i", fps);
			
				ImGui::BeginChild("Performance Region", ImVec2(0, 150), true);

				struct EventTiming {
					CUDAEvent::Desc desc;
					float           time;
				};
			
				int           event_timing_count = pathtracer.event_pool.num_used - 1;
				EventTiming * event_timings = new EventTiming[event_timing_count];

				for (int i = 0; i < event_timing_count; i++) {
					event_timings[i].desc = pathtracer.event_pool.pool[i].desc;
					event_timings[i].time = CUDAEvent::time_elapsed_between(
						pathtracer.event_pool.pool[i],
						pathtracer.event_pool.pool[i + 1]
					);
				}

				std::stable_sort(event_timings, event_timings + event_timing_count, [](const EventTiming & a, const EventTiming & b) {
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

							time_sum += event_timings[j].time;

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
					float time = 0.0f;

					while (true) {
						time += event_timings[i].time;

						if (i == event_timing_count - 1 || strcmp(event_timings[i].desc.name, event_timings[i+1].desc.name) != 0) break;

						i++;
					};

					ImGui::Text("%s: %*.2f ms", event_timings[i].desc.name, 5 + padding - strlen(event_timings[i].desc.name), time);

					if (i == event_timing_count - 1) {
						ImGui::TreePop();
						break;
					}

					category_changed = strcmp(event_timings[i].desc.category, event_timings[i + 1].desc.category);
					if (category_changed) {
						ImGui::TreePop();
					}
				}

				ImGui::EndChild();

				delete [] event_timings;
			}

			if (ImGui::CollapsingHeader("Settings", ImGuiTreeNodeFlags_DefaultOpen)) {
				bool invalidated_settings = false;
			
				invalidated_settings |= ImGui::SliderInt("Num Bounces", &pathtracer.settings.num_bounces, 0, MAX_BOUNCES);

				float fov = Math::rad_to_deg(pathtracer.scene.camera.fov);
				if (ImGui::SliderFloat("FOV", &fov, 0.0f, 179.0f)) {
					pathtracer.scene.camera.set_fov(Math::deg_to_rad(fov));
					pathtracer.invalidated_camera = true;
				}

				pathtracer.invalidated_camera |= ImGui::SliderFloat("Aperture", &pathtracer.scene.camera.aperture_radius,       0.0f,    1.0f);
				pathtracer.invalidated_camera |= ImGui::SliderFloat("Focus",    &pathtracer.scene.camera.focal_distance, 0.001f, 50.0f);

				invalidated_settings |= ImGui::Checkbox("NEE", &pathtracer.settings.enable_next_event_estimation);
				invalidated_settings |= ImGui::Checkbox("MIS", &pathtracer.settings.enable_multiple_importance_sampling);
			
				invalidated_settings |= ImGui::Checkbox("Update Scene", &pathtracer.settings.enable_scene_update);

				if (ImGui::Checkbox("SVGF", &pathtracer.settings.enable_svgf)) {
					if (pathtracer.settings.enable_svgf) {
						pathtracer.svgf_init();
					} else {
						pathtracer.svgf_free();
					}
					invalidated_settings = true;
				}

				invalidated_settings |= ImGui::Checkbox("Spatial Variance",  &pathtracer.settings.enable_spatial_variance);
				invalidated_settings |= ImGui::Checkbox("TAA",               &pathtracer.settings.enable_taa);
				invalidated_settings |= ImGui::Checkbox("Modulate Albedo",   &pathtracer.settings.modulate_albedo);

				invalidated_settings |= ImGui::Combo("Reconstruction Filter", reinterpret_cast<int *>(&pathtracer.settings.reconstruction_filter), "Box\0Gaussian\0");

				invalidated_settings |= ImGui::SliderInt("A Trous iterations", &pathtracer.settings.atrous_iterations, 0, MAX_ATROUS_ITERATIONS);

				invalidated_settings |= ImGui::SliderFloat("Alpha colour", &pathtracer.settings.alpha_colour, 0.0f, 1.0f);
				invalidated_settings |= ImGui::SliderFloat("Alpha moment", &pathtracer.settings.alpha_moment, 0.0f, 1.0f);

				pathtracer.invalidated_settings = invalidated_settings;
			}
		}
		ImGui::End();

		if (ImGui::Begin("Selected")) {
			if (Input::is_mouse_released(Input::MouseButton::RIGHT)) {
				// Deselect current object
				pathtracer.pixel_query.pixel_index = INVALID;
				pathtracer.pixel_query.mesh_id     = INVALID;
				pathtracer.pixel_query.triangle_id = INVALID;
				pathtracer.pixel_query.material_id = INVALID;
			}

			if (pathtracer.pixel_query.mesh_id != INVALID) {
				Mesh & mesh = pathtracer.scene.meshes[pathtracer.pixel_query.mesh_id];
				
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

				auto draw_line_clipped = [draw_list, &window](Vector4 a, Vector4 b, ImColor colour, float thickness = 1.0f) {
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
					const MeshData & mesh_data = pathtracer.scene.asset_manager.get_mesh_data(mesh.mesh_data_id);

					int              index    = mesh_data.bvh.indices[pathtracer.pixel_query.triangle_id - pathtracer.mesh_data_triangle_offsets[mesh.mesh_data_id.handle]];
					const Triangle & triangle = mesh_data.triangles[index];

					ImGui::Text("Distance: %f", Vector3::length(triangle.get_center() - pathtracer.scene.camera.position));

					Vector4 triangle_positions[3] = {
						Vector4(triangle.position_0.x, triangle.position_0.y, triangle.position_0.z, 1.0f),
						Vector4(triangle.position_1.x, triangle.position_1.y, triangle.position_1.z, 1.0f),
						Vector4(triangle.position_2.x, triangle.position_2.y, triangle.position_2.z, 1.0f)
					};

					for (int i = 0; i < 3; i++) {
						triangle_positions[i] = Matrix4::transform(pathtracer.scene.camera.view_projection * mesh.transform, triangle_positions[i]);
					}

					ImColor triangle_colour = ImColor(0.8f, 0.2f, 0.8f);

					draw_line_clipped(triangle_positions[0], triangle_positions[1], triangle_colour);
					draw_line_clipped(triangle_positions[1], triangle_positions[2], triangle_colour);
					draw_line_clipped(triangle_positions[2], triangle_positions[0], triangle_colour);
				}
			}

			if (pathtracer.pixel_query.material_id != INVALID) {
				Material & material = pathtracer.scene.asset_manager.get_material(MaterialHandle { pathtracer.pixel_query.material_id });

				ImGui::Separator();
				ImGui::Text("Name: %s", material.name);

				bool material_changed = false;

				int material_type = int(material.type);
				if (ImGui::Combo("Type", &material_type, "Light\0Diffuse\0Dielectric\0Glossy\0")) {
					material.type = Material::Type(material_type);
					material_changed = true;
				}

				switch (material.type) {
					case Material::Type::LIGHT: {
						material_changed |= ImGui::DragFloat3("Emission", &material.emission.x, 0.1f, 0.0f, INFINITY);
						break;
					}
					case Material::Type::DIFFUSE: {
						material_changed |= ImGui::SliderFloat3("Diffuse", &material.diffuse.x, 0.0f, 1.0f);
						material_changed |= ImGui::SliderInt   ("Texture", &material.texture_id.handle, -1, pathtracer.scene.asset_manager.textures.size() - 1);
						break;
					}
					case Material::Type::DIELECTRIC: {
						material_changed |= ImGui::SliderFloat3("Transmittance", &material.transmittance.x,     0.0f, 1.0f);
						material_changed |= ImGui::SliderFloat ("IOR",           &material.index_of_refraction, 1.0f, 5.0f);
						break;
					}
					case Material::Type::GLOSSY: {
						material_changed |= ImGui::SliderFloat3("Diffuse",   &material.diffuse.x, 0.0f, 1.0f);
						material_changed |= ImGui::SliderInt   ("Texture",   &material.texture_id.handle, -1, pathtracer.scene.asset_manager.textures.size() - 1);
						material_changed |= ImGui::SliderFloat ("IOR",       &material.index_of_refraction, 1.0f, 5.0f);
						material_changed |= ImGui::SliderFloat ("Roughness", &material.linear_roughness, 0.0f, 1.0f);
						break;
					}

					default: abort();
				}

				if (material_changed) pathtracer.invalidated_materials = true;
			}
		}
		ImGui::End();
		
		if (!ImGui::GetIO().WantCaptureMouse && Input::is_mouse_released()) {
			int mouse_x, mouse_y;			
			Input::mouse_position(&mouse_x, &mouse_y);

			pathtracer.set_pixel_query(mouse_x, mouse_y);
		}

		if (Input::is_key_released(SDL_SCANCODE_F5)) {
			ScopeTimer timer("Hot Reload");

			pathtracer.cuda_free();
			pathtracer.cuda_init(window.frame_buffer_handle, window.width, window.height);
		}

		if (perf_test.frame_end(delta_time)) break;

		Input::update(); // Save Keyboard State of this frame before SDL_PumpEvents

		window.gui_end();
		window.swap();
	}

	CUDAContext::destroy();

	return EXIT_SUCCESS;
}
