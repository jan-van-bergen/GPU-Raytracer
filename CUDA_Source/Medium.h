#pragma once

struct Medium {
    float3 sigma_a;
    float3 sigma_s;
    float  g;
};

__device__ __constant__ Medium * mediums;

__device__ inline float3 beer_lambert(const float3 & sigma_t, float distance) {
	return make_float3(
		expf(-sigma_t.x * distance),
		expf(-sigma_t.y * distance),
		expf(-sigma_t.z * distance)
	);
}
