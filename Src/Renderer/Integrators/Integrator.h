#pragma once
#include "Core/Random.h"
#include "CUDA/Common.h"

#include "Device/CUDAModule.h"
#include "Device/CUDAKernel.h"
#include "Device/CUDAMemory.h"
#include "Device/CUDAEvent.h"
#include "Device/CUDAContext.h"

#include "BVH/Builders/SAHBuilder.h"
#include "BVH/Converters/BVHConverter.h"

#include "Renderer/Scene.h"

#include "Util/PMJ.h"


// Mirror CUDA vector types
struct alignas(8)  float2 { float x, y; };
struct             float3 { float x, y, z; };
struct alignas(16) float4 { float x, y, z, w; };

struct alignas(8)  int2 { int x, y; };
struct             int3 { int x, y, z; };
struct alignas(16) int4 { int x, y, z, w; };

struct Matrix3x4 {
	float cells[12];
};

struct CUDAVector3_SoA {
	CUDAMemory::Ptr<float> x;
	CUDAMemory::Ptr<float> y;
	CUDAMemory::Ptr<float> z;

	inline void init(int buffer_size) {
		x = CUDAMemory::malloc<float>(buffer_size);
		y = CUDAMemory::malloc<float>(buffer_size);
		z = CUDAMemory::malloc<float>(buffer_size);
	}

	inline void free() {
		CUDAMemory::free(x);
		CUDAMemory::free(y);
		CUDAMemory::free(z);
	}
};

// Arbitrary Output Variable
struct AOV {
	CUDAMemory::Ptr<float4> framebuffer;
	CUDAMemory::Ptr<float4> accumulator;
};

struct Integrator {
	Scene & scene;

	bool invalidated_scene      = true;
	bool invalidated_sky        = true;
	bool invalidated_materials  = true;
	bool invalidated_mediums    = true;
	bool invalidated_camera     = true;
	bool invalidated_gpu_config = true;
	bool invalidated_aovs       = true;

	int screen_width;
	int screen_height;
	int screen_pitch;

	int pixel_count;

	int sample_index = 0;

	enum struct PixelQueryStatus {
		INACTIVE,
		PENDING,
		OUTPUT_READY
	} pixel_query_status = PixelQueryStatus::INACTIVE;

	PixelQuery pixel_query = { INVALID, INVALID, INVALID };

	CUDAModule::Global global_pixel_query;

	CUDAModule cuda_module;

	CUstream memory_stream = { };

	CUgraphicsResource resource_accumulator;
	CUsurfObject       surf_accumulator;

	union alignas(float4) CUDAMaterial {
		struct {
			Vector3 emission;
		} light;
		struct {
			Vector3 diffuse;
			int     texture_id;
		} diffuse;
		struct {
			Vector3 diffuse;
			int     texture_id;
			float   roughness;
		} plastic;
		struct {
			int   medium_id;
			float ior;
			float roughness;
		} dielectric;
		struct {
			Vector3 eta;
			float   roughness;
			Vector3 k;
		} conductor;

		CUDAMaterial() { }
	};

	CUDAMemory::Ptr<Material::Type> ptr_material_types;
	CUDAMemory::Ptr<CUDAMaterial>   ptr_materials;

	struct alignas(float4) CUDAMedium {
		Vector3 sigma_a;
		float   g;
		Vector3 sigma_s;
	};
	CUDAMemory::Ptr<CUDAMedium> ptr_media;

	struct CUDATexture {
		CUtexObject texture;
		float       lod_bias;
	};

	Array<CUDATexture>      textures;
	Array<CUmipmappedArray> texture_arrays;

	CUDAMemory::Ptr<CUDATexture> ptr_textures;

	struct CUDATriangle {
		Vector3 position_0;
		Vector3 position_edge_1;
		Vector3 position_edge_2;

		Vector3 normal_0;
		Vector3 normal_edge_1;
		Vector3 normal_edge_2;

		Vector2 tex_coord_0;
		Vector2 tex_coord_edge_1;
		Vector2 tex_coord_edge_2;
	};

	CUDAMemory::Ptr<CUDATriangle> ptr_triangles;

	CUDAMemory::Ptr<BVHNode2>  ptr_bvh_nodes_2;
	CUDAMemory::Ptr<BVHNode4>  ptr_bvh_nodes_4;
	CUDAMemory::Ptr<BVHNode8>  ptr_bvh_nodes_8;
	CUDAMemory::Ptr<int>       ptr_mesh_bvh_root_indices;
	CUDAMemory::Ptr<int>       ptr_mesh_material_ids;
	CUDAMemory::Ptr<Matrix3x4> ptr_mesh_transforms;
	CUDAMemory::Ptr<Matrix3x4> ptr_mesh_transforms_inv;
	CUDAMemory::Ptr<Matrix3x4> ptr_mesh_transforms_prev;

	int       * pinned_mesh_bvh_root_indices             = nullptr;
	int       * pinned_mesh_material_ids                 = nullptr;
	Matrix3x4 * pinned_mesh_transforms                   = nullptr;
	Matrix3x4 * pinned_mesh_transforms_inv               = nullptr;
	Matrix3x4 * pinned_mesh_transforms_prev              = nullptr;
	float     * pinned_light_mesh_cumulative_probability = nullptr;
	int2      * pinned_light_mesh_triangle_span          = nullptr;
	int       * pinned_light_mesh_transform_indices      = nullptr;

	BVH2                 tlas_raw;
	OwnPtr<BVH>          tlas;
	OwnPtr<SAHBuilder>   tlas_builder;
	OwnPtr<BVHConverter> tlas_converter;

	Array<int> reverse_indices;

	Array<int> mesh_data_bvh_offsets;
	Array<int> mesh_data_triangle_offsets;

	CUDAModule::Global global_camera;
	CUDAModule::Global global_sky_scale;
	CUDAModule::Global global_config;
	CUDAModule::Global global_buffer_sizes;

	CUDAMemory::Ptr<Vector3> ptr_sky_data;

	CUDAMemory::Ptr<PMJ::Point>     ptr_pmj_samples;
	CUDAMemory::Ptr<unsigned short> ptr_blue_noise_textures;

	CUDAEventPool event_pool;

	AOV aovs[size_t(AOVType::COUNT)];
	CUDAModule::Global global_aovs;

	Integrator(Scene & scene) : scene(scene) {
		CUDACALL(cuStreamCreate(&memory_stream, CU_STREAM_NON_BLOCKING));
	}

	virtual void cuda_init(unsigned frame_buffer_handle, int screen_width, int screen_height) {
		scene.camera.update(0.0f);
		scene.update(0.0f);

		scene.has_diffuse    = false;
		scene.has_plastic    = false;
		scene.has_dielectric = false;
		scene.has_conductor  = false;
		scene.has_lights     = false;

		invalidated_scene      = true;
		invalidated_sky        = true;
		invalidated_materials  = true;
		invalidated_mediums    = true;
		invalidated_gpu_config = true;
		invalidated_aovs       = true;

		size_t bytes_available = CUDAContext::get_available_memory();
		size_t bytes_allocated = CUDAContext::total_memory - bytes_available;
		IO::print("CUDA Memory allocated: {} KB ({} MB)\n"_sv,   bytes_allocated >> 10, bytes_allocated >> 20);
		IO::print("CUDA Memory free:      {} KB ({} MB)\n\n"_sv, bytes_available >> 10, bytes_available >> 20);
	}

	virtual void cuda_free() {
		resize_free();
		cuda_module.free();
	}

	void init_globals();
	void init_materials();
	void init_geometry();
	void init_sky();
	void init_rng();
	void init_aovs();

	void free_materials();
	void free_geometry();
	void free_sky();
	void free_rng();
	void free_aovs();

	virtual void resize_free() = 0;
	virtual void resize_init(unsigned frame_buffer_handle, int width, int height) = 0;

	      AOV & get_aov(AOVType aov_type)       { return aovs[size_t(aov_type)]; }
	const AOV & get_aov(AOVType aov_type) const { return aovs[size_t(aov_type)]; }

	void aov_enable (AOVType aov_type) { gpu_config.aov_mask |=  (1u << int(aov_type)); invalidated_aovs = true; }
	void aov_disable(AOVType aov_type) { gpu_config.aov_mask &= ~(1u << int(aov_type)); invalidated_aovs = true; }

	bool aov_is_enabled(AOVType aov_type) const { return gpu_config.aov_mask & (1u << int(aov_type)); }

	void aovs_clear_to_zero();

	bool aov_render_gui_checkbox(AOVType aov_type, const char * aov_name);

	void build_tlas();

	virtual void update(float delta, Allocator * frame_allocator);
	virtual void render() = 0;

	virtual void render_gui() = 0;

	void set_pixel_query(int x, int y) {
		if (x < 0 || y < 0 || x >= screen_width || y >= screen_height) return;

		y = screen_height - y; // Y-coordinate is inverted

		pixel_query.pixel_index = x + y * screen_pitch;
		pixel_query.mesh_id     = INVALID;
		pixel_query.triangle_id = INVALID;
		global_pixel_query.set_value_async(pixel_query, memory_stream);

		pixel_query_status = PixelQueryStatus::PENDING;
	}

	// Helper method to calculate the optimal launch bounds for the BVH traversal kernels
	template<size_t BVH_STACK_ELEMENT_SIZE>
	void kernel_trace_calc_grid_and_block_size(CUDAKernel & kernel) {
		CUoccupancyB2DSize block_size_to_shared_memory = [](int block_size) {
			return size_t(block_size) * SHARED_STACK_SIZE * BVH_STACK_ELEMENT_SIZE;
		};

		int grid, block;
		CUDACALL(cuOccupancyMaxPotentialBlockSize(&grid, &block, kernel.kernel, block_size_to_shared_memory, 0, 0));

		int block_x = WARP_SIZE;
		int block_y = block / WARP_SIZE;

		kernel.set_block_dim(block_x, block_y, 1);
		kernel.set_grid_dim(1, grid, 1);
		kernel.set_shared_memory(block_size_to_shared_memory(block));
	};
};
