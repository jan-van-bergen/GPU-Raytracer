#pragma once
// This file contains things that are shared between the CUDA files and the C++ files


// Constants
#define PI          3.14159265359f
#define ONE_OVER_PI 0.31830988618f

#define TWO_PI          6.28318530718f
#define ONE_OVER_TWO_PI 0.15915494309f


// CUDA
#define WARP_SIZE 32

#define MAX_REGISTERS 64


// Settings
enum class ReconstructionFilter {
	BOX,
	MITCHELL_NETRAVALI,
	GAUSSIAN
};

struct Settings {
	bool enable_rasterization                = false;
	bool enable_next_event_estimation        = true;
	bool enable_multiple_importance_sampling = true;
	bool enable_scene_update                 = false;
	bool enable_svgf                         = false;
	bool enable_spatial_variance             = true;
	bool enable_taa                          = true;
	
	bool demodulate_albedo = false;

	ReconstructionFilter reconstruction_filter = ReconstructionFilter::BOX;

	// SVGF Settings
	float alpha_colour = 0.2f;
	float alpha_moment = 0.2f;

	int atrous_iterations = 5;

	float sigma_z =  4.0f;
	float sigma_n = 16.0f;
	float sigma_l = 10.0f;
};


// Screen related
#define SCREEN_WIDTH  900
#define SCREEN_HEIGHT 600


// Rendering is performance in batches of BATCH_SIZE pixels
// Larger batches are more efficient, but also require more GPU memory
#define BATCH_SIZE (SCREEN_WIDTH * SCREEN_HEIGHT)

// Raytracing
#define EPSILON 0.001f

#define NUM_BOUNCES 5


// Lighting
#define LIGHT_SELECT_UNIFORM 0
#define LIGHT_SELECT_AREA    1

#define LIGHT_SELECTION LIGHT_SELECT_AREA


// SVGF
#define MAX_ATROUS_ITERATIONS 10


// Mipmapping
#define ENABLE_MIPMAPPING true


// Microfacet
#define MICROFACET_BECKMANN 0 
#define MICROFACET_GGX      1

#define MICROFACET_MODEL MICROFACET_GGX

// If MICROFACET_SEPARATE_G_TERMS is set to true, two monodirectional Smith terms G1 are used,
// otherwise a Height-Correlated Masking and Shadowing term G2 is used based on 2 lambda terms.
#define MICROFACET_SEPARATE_G_TERMS false


// BVH related
#define BVH_BVH   0 // Binary SAH-based BVH
#define BVH_SBVH  1 // Binary SAH-based Spatial BVH
#define BVH_QBVH  2 // Quaternary BVH, constructed by collapsing the binary SBVH
#define BVH_CWBVH 3 // Compressed Wide BVH (8 way)

#define BVH_TYPE BVH_CWBVH

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
