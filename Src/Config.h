#pragma once
#include "CUDA/Common.h"

#include "Core/Array.h"
#include "Core/String.h"

inline GPUConfig gpu_config = { };

enum struct MipmapFilterType {
	BOX,
	LANCZOS,
	KAISER
};

enum struct BVHType {
	BVH,  // Binary SAH-based BVH
	SBVH, // Binary SAH-based Spatial BVH
	BVH4, // Quaternary BVH,              constructed by collapsing the binary BVH
	BVH8  // Compressed Wide BVH (8 way), constructed by collapsing the binary BVH
};

struct CPUConfig {
	int initial_width  = 900;
	int initial_height = 600;

	Array<String> scene_filenames;
	String        sky_filename;

	int    output_sample_index = INVALID;
	String output_name         = "render.ppm"_sv;

	bool bvh_force_rebuild        = false;
	bool enable_bvh_optimization  = false;
	bool enable_block_compression = true;
	bool enable_scene_update      = false;

	MipmapFilterType mipmap_filter = MipmapFilterType::BOX;

	BVHType bvh_type = BVHType::BVH8;

	float sah_cost_node = 4.0f;
	float sah_cost_leaf = 1.0f;

	float sbvh_alpha = 10e-5f; // Alpha parameter for SBVH construction, alpha == 1 means regular BVH, alpha == 0 means full SBVH

	int bvh_optimizer_max_time        = 60000; // Time limit in milliseconds
	int bvh_optimizer_max_num_batches = 1000;
};

inline CPUConfig cpu_config = { };
