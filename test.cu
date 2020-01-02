#include "vector_types.h"

surface<void, 2> output_surface;

#define SCREEN_WIDTH  800
#define SCREEN_HEIGHT 480

extern "C" __global__ void test_function() {
	// get x and y for pixel
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= SCREEN_WIDTH) || (y >= SCREEN_HEIGHT)) return;

	float r = (float)x / SCREEN_WIDTH;
	float g = (float)y / SCREEN_HEIGHT;

	surf2Dwrite<float4>(make_float4(r, g, 0.0f, 1.0f), output_surface, x * sizeof(float4), y, cudaBoundaryModeClamp);
}
