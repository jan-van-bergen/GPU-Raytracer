#pragma once

struct Medium {
    float3 scatter_coefficient;
    float3 negative_absorption;
};

__device__ __constant__ Medium * mediums;

__device__ inline float3 beer_lambert(const float3 & negative_absorption, float distance) {
    return make_float3(
        expf(negative_absorption.x * distance),
		expf(negative_absorption.y * distance),
		expf(negative_absorption.z * distance)
    );
}
