#pragma once
#include "../CUDA_Source/Common.h"
#include "Util/Util.h"

enum struct BVHType {
	BVH,  // Binary SAH-based BVH
	SBVH, // Binary SAH-based Spatial BVH
	QBVH, // Quaternary BVH,              constructed by collapsing the binary BVH
	CWBVH // Compressed Wide BVH (8 way), constructed by collapsing the binary BVH
};

struct Config {
	// Screen
	int initial_width  = 900;
	int initial_height = 600;


	// Scene
	const char * scene = DATA_PATH("sponza/scene.xml");
	const char * sky   = DATA_PATH("Sky_Probes/sky_15.hdr");


	// GPU shared settings
	Settings settings;

	int capture_frame_index = INVALID;


	// Textures
	bool enable_block_compression = true;

	enum struct MipmapFilter {
		BOX,
		LANCZOS,
		KAISER
	} mipmap_filter = MipmapFilter::KAISER;


	// BVH
	BVHType bvh_type = BVHType::CWBVH;

	float sah_cost_node = 4.0f;
	float sah_cost_leaf = 1.0f;

	float sbvh_alpha = 10e-5f; // Alpha parameter for SBVH construction, alpha == 1 means regular BVH, alpha == 0 means full SBVH

	bool enable_bvh_optimization = true;

	int bvh_optimizer_max_time        = -1; // Time limit in milliseconds
	int bvh_optimizer_max_num_batches = 1000;
};

inline Config config = { };
