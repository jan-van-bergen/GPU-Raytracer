#pragma once
#include <vector_types.h>

#include "../Common.h"

__device__ int      sky_size;
__device__ float3 * sky_data;

__device__ float3 sample_sky(const float3 & direction) {
	// Formulas as described on https://www.pauldebevec.com/Probes/
    float r = 0.5f * ONE_OVER_PI * acos(direction.z) * rsqrt(direction.x*direction.x + direction.y*direction.y);

	float u = direction.x * r + 0.5f;
	float v = direction.y * r + 0.5f;

	// Convert to pixel coordinates
	int x = int(u * sky_size);
	int y = int(v * sky_size);

	int index = x + y * sky_size;
	index = max(index, 0);
	index = min(index, sky_size * sky_size);

	return sky_data[index];
}
