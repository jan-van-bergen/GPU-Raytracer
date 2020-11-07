#pragma once

__device__ __constant__ int      sky_width;
__device__ __constant__ int      sky_height;
__device__ __constant__ float3 * sky_data;

__device__ float3 sample_sky(const float3 & direction) {
	// Convert direction to spherical coordinates
	float phi   = atan2f(-direction.z, direction.x);
	float theta = acosf(clamp(direction.y, -1.0f, 1.0f));

	float u = phi   * ONE_OVER_TWO_PI;
	float v = theta * ONE_OVER_PI;

	// Convert to pixel coordinates
	int x = int(u * sky_width);
	int y = int(v * sky_height);

	int index = clamp(x + y * sky_width, 0, sky_width * sky_height - 1);

	return sky_data[index];
}
