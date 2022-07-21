#pragma once
#include "Util.h"

__device__ __constant__ Texture<float4> sky_texture;
__device__ __constant__ float           sky_scale;

__device__ float3 sample_sky(float3 direction) {
	// Convert direction to spherical coordinates
	float phi   = atan2f(-direction.z, direction.x);
	float theta = acosf(clamp(direction.y, -1.0f, 1.0f));

	float u = phi   * ONE_OVER_TWO_PI + 0.5f;
	float v = theta * ONE_OVER_PI;

	return sky_scale * make_float3(sky_texture.get(u, v));
}
