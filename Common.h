#pragma once
// This file contains things that are shared between the CUDA files and the C++ files

#define SCREEN_WIDTH  800
#define SCREEN_HEIGHT 480

#define PIXEL_COUNT (SCREEN_WIDTH * SCREEN_HEIGHT)

#define EPSILON 0.001f

#define PI          3.14159265359f
#define ONE_OVER_PI 0.31830988618f

#define TWO_PI          6.28318530718f
#define ONE_OVER_TWO_PI 0.15915494309f

#define NUM_BOUNCES 5

#define MAX_MATERIALS 32
#define MAX_TEXTURES  32

#define BVH_TRAVERSE_TREE_NAIVE   0 // Traverses the BVH in a naive way, always checking the left Node before the right Node
#define BVH_TRAVERSE_TREE_ORDERED 1 // Traverses the BVH based on the split axis and the direction of the Ray

#define BVH_TRAVERSAL_STRATEGY BVH_TRAVERSE_TREE_ORDERED

#define BVH_AXIS_X_BITS 0x40000000 // 01 00 zeroes...
#define BVH_AXIS_Y_BITS 0x80000000 // 10 00 zeroes...
#define BVH_AXIS_Z_BITS 0xc0000000 // 11 00 zeroes...
#define BVH_AXIS_MASK   0xc0000000 // 11 00 zeroes...

#define MBVH_WIDTH 4
static_assert(MBVH_WIDTH >= 2, "MBVH tree must be at least binary");
