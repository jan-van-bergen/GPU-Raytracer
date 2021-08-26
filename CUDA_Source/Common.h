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
	GAUSSIAN
};

struct Settings {
	int num_bounces = 5;

	bool enable_next_event_estimation        = true;
	bool enable_multiple_importance_sampling = true;
	bool enable_scene_update                 = false;
	bool enable_svgf                         = false;
	bool enable_spatial_variance             = true;
	bool enable_taa                          = true;

	bool modulate_albedo = true;

	ReconstructionFilter reconstruction_filter = ReconstructionFilter::GAUSSIAN;

	// SVGF Settings
	float alpha_colour = 0.1f;
	float alpha_moment = 0.1f;

	int atrous_iterations = 4;

	float sigma_z =  4.0f;
	float sigma_n = 16.0f;
	float sigma_l = 10.0f;
};


// Screen size at startup
#define SCREEN_WIDTH  900
#define SCREEN_HEIGHT 600


// Rendering is performance in batches of BATCH_SIZE pixels
// Larger batches are more efficient, but also require more GPU memory
#define BATCH_SIZE (SCREEN_WIDTH * SCREEN_HEIGHT)

// Raytracing
#define EPSILON 0.001f

#define MAX_BOUNCES 20


// SVGF
#define MAX_ATROUS_ITERATIONS 10


// Textures
#define ENABLE_BLOCK_COMPRESSION true

#define MIPMAP_DOWNSAMPLE_FILTER_BOX     0
#define MIPMAP_DOWNSAMPLE_FILTER_LANCZOS 1
#define MIPMAP_DOWNSAMPLE_FILTER_KAISER  2

#define MIPMAP_DOWNSAMPLE_FILTER MIPMAP_DOWNSAMPLE_FILTER_KAISER

#define ENABLE_MIPMAPPING true


// BVH related
#define SAH_COST_NODE 4.0f
#define SAH_COST_LEAF 1.0f

#define BVH_BVH   0 // Binary SAH-based BVH
#define BVH_SBVH  1 // Binary SAH-based Spatial BVH
#define BVH_QBVH  2 // Quaternary BVH,              constructed by collapsing the binary BVH
#define BVH_CWBVH 3 // Compressed Wide BVH (8 way), constructed by collapsing the binary BVH

#define BVH_TYPE BVH_CWBVH

#define SBVH_ALPHA 10e-5f // Alpha parameter for SBVH construction, alpha == 1 means regular BVH, alpha == 0 means full SBVH

#define BVH_ENABLE_OPTIMIZATION true

#define BVH_OPTIMIZER_MAX_TIME -1 // Time limit in milliseconds
#define BVH_OPTIMIZER_MAX_NUM_BATCHES 1000


// Inverse of the percentage of active threads that triggers triangle postponing
// A value of 5 means that if less than 1/5 = 20% of the active threads want to
// intersect triangles we postpone the intersection test to decrease divergence within a Warp
#define CWBVH_TRIANGLE_POSTPONING_THRESHOLD_DIVISOR 5

#define BVH_STACK_SIZE 32

// Portion of the Stack that resides in Shared Memory
#define SHARED_STACK_SIZE 8
static_assert(SHARED_STACK_SIZE < BVH_STACK_SIZE, "Shared Stack size must be strictly smaller than total Stack size");


#define BVH_AXIS_X_BITS (0b01 << 30)
#define BVH_AXIS_Y_BITS (0b10 << 30)
#define BVH_AXIS_Z_BITS (0b11 << 30)
#define BVH_AXIS_MASK   (0b11 << 30)


// Used to perform mouse interaction with objects in the scene
struct PixelQuery {
	int pixel_index; // x + y * screen_pitch

	int mesh_id;
	int triangle_id;
	int material_id;
};
