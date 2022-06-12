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

// Arbitrary Output Variables
enum struct AOVType {
    RADIANCE,
    RADIANCE_DIRECT,
    RADIANCE_INDIRECT,
    ALBEDO,
    NORMAL,
    POSITION,

    COUNT
};

struct GPUConfig {
	// Output
	ReconstructionFilter reconstruction_filter = ReconstructionFilter::GAUSSIAN;

	unsigned aov_mask = 0;


	// Pathtracing
	int num_bounces = 10;

	bool enable_mipmapping                   = true;
	bool enable_next_event_estimation        = true;
	bool enable_multiple_importance_sampling = true;
	bool enable_russian_roulette             = true;
	bool enable_svgf                         = false;
	bool enable_spatial_variance             = true;
	bool enable_taa                          = true;


	// SVGF
	float alpha_colour = 0.1f;
	float alpha_moment = 0.1f;

	int num_atrous_iterations = 6;

	float sigma_z =  4.0f;
	float sigma_n = 16.0f;
	float sigma_l = 10.0f;
};

// Rendering is performance in batches of BATCH_SIZE pixels
// Larger batches are more efficient, but also require more GPU memory
#define BATCH_SIZE (1080 * 720)


// Raytracing
#define EPSILON 0.0001f
#define MAX_BOUNCES 128


// RNG
#define PMJ_NUM_SEQUENCES 64
#define PMJ_NUM_SAMPLES_PER_SEQUENCE 4096

#define BLUE_NOISE_NUM_TEXTURES 16
#define BLUE_NOISE_TEXTURE_DIM 128


// BSDF LUTS
#define LUT_DIELECTRIC_DIM_IOR 16
#define LUT_DIELECTRIC_DIM_ROUGHNESS 16
#define LUT_DIELECTRIC_DIM_COS_THETA 16

#define LUT_DIELECTRIC_MIN_IOR 1.0001f
#define LUT_DIELECTRIC_MAX_IOR 2.5f

#define LUT_CONDUCTOR_DIM_ROUGHNESS 32
#define LUT_CONDUCTOR_DIM_COS_THETA 32


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
};
