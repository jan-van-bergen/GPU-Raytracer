#pragma once
// This file contains things that are shared between the CUDA files and the C++ files

#define WARP_SIZE 32


// Screen related
#define SCREEN_WIDTH  904
#define SCREEN_HEIGHT 600

#define PIXEL_COUNT (SCREEN_WIDTH * SCREEN_HEIGHT)

#define SCREEN_PITCH ((SCREEN_WIDTH + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE)


#define BLOCK_WIDTH  8
#define BLOCK_HEIGHT 4
#define BLOCK_SIZE   (BLOCK_WIDTH * BLOCK_HEIGHT)

static_assert(SCREEN_WIDTH  % BLOCK_WIDTH  == 0, "Screen width  should be divisible by the Block width!");
static_assert(SCREEN_HEIGHT % BLOCK_HEIGHT == 0, "Screen height should be divisible by the Block height!");
static_assert(BLOCK_SIZE == WARP_SIZE, "Block size should equal CUDA warp size!");


// Tracing related
#define EPSILON 0.001f

#define NUM_BOUNCES 5


#define        TRACE_BLOCK_Y 2
#define SHADOW_TRACE_BLOCK_Y 2


// SVGF
#define MAX_ATROUS_ITERATIONS 10


// Pi related
#define PI          3.14159265359f
#define ONE_OVER_PI 0.31830988618f

#define TWO_PI          6.28318530718f
#define ONE_OVER_TWO_PI 0.15915494309f


// BVH related
#define BVH_BVH   0 // Binary SAH-based BVH
#define BVH_SBVH  1 // Binary SAH-based Spatial BVH
#define BVH_QBVH  2 // Quaternary BVH, constructed by collapsing the binary SBVH
#define BVH_CWBVH 3 // Compressed Wide BVH (8 way)

#define BVH_TYPE BVH_CWBVH

#define SBVH_MAX_PRIMITIVES_IN_LEAF 1

#define BVH_STACK_SIZE 32

// Portion of the Stack that resides in Shared Memory
#define SHARED_STACK_SIZE 8
static_assert(SHARED_STACK_SIZE < BVH_STACK_SIZE, "Shared Stack size must be strictly smaller than total Stack size");


#define BVH_AXIS_X_BITS (0b01 << 30)
#define BVH_AXIS_Y_BITS (0b10 << 30)
#define BVH_AXIS_Z_BITS (0b11 << 30)
#define BVH_AXIS_MASK   (0b11 << 30)
