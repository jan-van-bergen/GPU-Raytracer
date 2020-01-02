#include "vector_types.h"

#include "Common.h"

surface<void, 2> output_surface;

//texture<float, 1, cudaReadModeElementType> triangle_texture;

extern "C" __global__ void test_function(float a, float b) {
	// get x and y for pixel
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	float r = float(x) / SCREEN_WIDTH;
	float g = float(y) / SCREEN_HEIGHT;

	surf2Dwrite<float4>(make_float4(r, g, 0.0f, 1.0f), output_surface, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

extern "C" __global__ void trace_ray() {
	float x = float(threadIdx.x + blockIdx.x * blockDim.x) / SCREEN_WIDTH;
	float y = float(threadIdx.y + blockIdx.y * blockDim.y) / SCREEN_HEIGHT;
}
