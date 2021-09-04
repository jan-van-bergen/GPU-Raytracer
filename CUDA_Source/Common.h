#pragma once
// This file contains things that are shared between the CUDA files and the C++ files


// Constants
#define PI          3.14159265359f
#define ONE_OVER_PI 0.31830988618f

#define TWO_PI          6.28318530718f
#define ONE_OVER_TWO_PI 0.15915494309f

#define INVALID -1


// CUDA
#define WARP_SIZE 32
#define MAX_REGISTERS 64


// Settings
enum struct ReconstructionFilter {
	BOX,
	TENT,
	GAUSSIAN
};

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

	ReconstructionFilter reconstruction_filter = ReconstructionFilter::GAUSSIAN;


	// Scene
	const char * scene = "Data/cornellbox/scene.xml";
	const char * sky   = "Data/Skies/sky_15.hdr";


	// Pathtracing
	int num_bounces = 5;

	int capture_frame_index = INVALID;

	bool enable_albedo                       = true;
	bool enable_mipmapping                   = true;
	bool enable_next_event_estimation        = true;
	bool enable_multiple_importance_sampling = true;
	bool enable_russian_roulette             = true;
	bool enable_scene_update                 = false;
	bool enable_svgf                         = false;
	bool enable_spatial_variance             = true;
	bool enable_taa                          = true;


	// SVGF
	float alpha_colour = 0.1f;
	float alpha_moment = 0.1f;

	int num_atrous_iterations = 4;

	float sigma_z =  4.0f;
	float sigma_n = 16.0f;
	float sigma_l = 10.0f;


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

// Rendering is performance in batches of BATCH_SIZE pixels
// Larger batches are more efficient, but also require more GPU memory
#define BATCH_SIZE (1080 * 720)


// Raytracing
#define EPSILON 0.001f
#define MAX_BOUNCES 128


// PMJ
#define PMJ_NUM_SEQUENCES 64
#define PMJ_NUM_SAMPLES_PER_SEQUENCE 4096


// SVGF
#define MAX_ATROUS_ITERATIONS 10


// BVH
#define BVH_STACK_SIZE 32

// Portion of the Stack that resides in Shared Memory
#define SHARED_STACK_SIZE 8
static_assert(SHARED_STACK_SIZE < BVH_STACK_SIZE, "Shared Stack size must be strictly smaller than total Stack size");


// Used to perform mouse interaction with objects in the scene
struct PixelQuery {
	int pixel_index; // x + y * screen_pitch

	int mesh_id;
	int triangle_id;
	int material_id;
};
