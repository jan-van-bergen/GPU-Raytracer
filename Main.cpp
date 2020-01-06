#include <cstdio>
#include <cstdlib>
#include <time.h> 

#include "Window.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"
#include "CUDAMemory.h"
#include "CUDAContext.h"

#include "Camera.h"

#include "MeshData.h"
#include "BVH.h"

// Forces NVIDIA driver to be used 
extern "C" { _declspec(dllexport) unsigned NvOptimusEnablement = true; }

#define TOTAL_TIMING_COUNT 1000
float timings[TOTAL_TIMING_COUNT];
int   current_frame = 0;

int main(int argument_count, char ** arguments) {
	Window window(SCREEN_WIDTH, SCREEN_HEIGHT, "Pathtracer");

	CUDAContext::init();

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
	module.init("Pathtracer.cu", CUDAContext::compute_capability);

	const MeshData * mesh = MeshData::load(DATA_PATH("Scene.obj"));

	if (mesh->material_count > MAX_MATERIALS || Texture::texture_count > MAX_TEXTURES) abort();

	// Set global Material table
	CUdeviceptr materials_ptr = CUDAMemory::malloc<Material>(mesh->material_count);
	CUDAMemory::memcpy(materials_ptr, mesh->materials, mesh->material_count);

	module.get_global("materials").set(materials_ptr);

	// Set global Texture table
	if (Texture::texture_count > 0) {
		CUtexObject * tex_objects = new CUtexObject[Texture::texture_count];

		for (int i = 0; i < Texture::texture_count; i++) {
			CUarray array = CUDAMemory::create_array(Texture::textures[i].width, Texture::textures[i].height, Texture::textures[i].channels, CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8);
		
			CUDAMemory::copy_array(array, Texture::textures[i].channels * Texture::textures[i].width, Texture::textures[i].height, Texture::textures[i].data);

			// Describe the Array to read from
			CUDA_RESOURCE_DESC res_desc = { };
			res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
			res_desc.res.array.hArray = array;
		
			// Describe how to sample the Texture
			CUDA_TEXTURE_DESC tex_desc = { };
			tex_desc.addressMode[0] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.addressMode[1] = CUaddress_mode::CU_TR_ADDRESS_MODE_WRAP;
			tex_desc.filterMode = CUfilter_mode::CU_TR_FILTER_MODE_LINEAR;
			tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
			
			CUDACALL(cuTexObjectCreate(tex_objects + i, &res_desc, &tex_desc, nullptr));
		}

		CUdeviceptr textures_ptr = CUDAMemory::malloc<CUtexObject>(Texture::texture_count);
		CUDAMemory::memcpy(textures_ptr, tex_objects, Texture::texture_count);

		module.get_global("textures").set(textures_ptr);
	}

	BVH<Triangle> bvh;
	bvh.init(mesh->triangle_count);
	
	memcpy(bvh.primitives, mesh->triangles, mesh->triangle_count * sizeof(Triangle));

	for (int i = 0; i < bvh.primitive_count; i++) {
		Vector3 vertices[3] = { 
			bvh.primitives[i].position0, 
			bvh.primitives[i].position1, 
			bvh.primitives[i].position2
		};
		bvh.primitives[i].aabb = AABB::from_points(vertices, 3);
	}

	bvh.build_bvh();

	// Set global Triangle buffer
	CUdeviceptr triangles_ptr = CUDAMemory::malloc<Triangle>(bvh.primitive_count);
	CUDAMemory::memcpy(triangles_ptr, bvh.primitives, bvh.primitive_count);

	module.get_global("triangles").set(triangles_ptr);

	// Set global BVHNode buffer
	CUdeviceptr nodes_ptr = CUDAMemory::malloc<BVHNode>(bvh.node_count);
	CUDAMemory::memcpy(nodes_ptr, bvh.nodes, bvh.node_count);

	module.get_global("bvh_nodes").set(nodes_ptr);

	// Set Camera globals
	CUDAModule::Global global_camera_position        = module.get_global("camera_position");
	CUDAModule::Global global_camera_top_left_corner = module.get_global("camera_top_left_corner");
	CUDAModule::Global global_camera_x_axis          = module.get_global("camera_x_axis");
	CUDAModule::Global global_camera_y_axis          = module.get_global("camera_y_axis");
	
	// Set frame buffer to a CUDA resource mapping of the GL frame buffer texture
	module.set_surface("frame_buffer", CUDAContext::map_gl_texture(window.frame_buffer_handle));

	// Initialize Kernel
	CUDAKernel kernel;
	kernel.init(&module, "trace_ray");

	kernel.set_block_dim(32, 4, 1);
	kernel.set_grid_dim(SCREEN_WIDTH / kernel.block_dim_x, SCREEN_HEIGHT / kernel.block_dim_y, 1);

	last = SDL_GetPerformanceCounter();

	srand(time(nullptr));

	float frames_since_last_camera_moved = 0.0f;

	// Game loop
	while (!window.is_closed) {
		camera.update(delta_time, SDL_GetKeyboardState(nullptr));
		
		global_camera_position.set(camera.position);
		global_camera_top_left_corner.set(camera.top_left_corner_rotated);
		global_camera_x_axis.set(camera.x_axis_rotated);
		global_camera_y_axis.set(camera.y_axis_rotated);

		if (camera.moved) {
			frames_since_last_camera_moved = 0.0f;
		} else {
			frames_since_last_camera_moved += 1.0f;
		}

		kernel.execute(rand(), frames_since_last_camera_moved);

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
