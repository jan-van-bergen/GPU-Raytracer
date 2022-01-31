#pragma once
#include "Common.h"

struct AOV {
    float4 * framebuffer; // Written to on the current frame
    float4 * accumulator; // Accumulated over N frames
};

__device__ __constant__ AOV aovs[size_t(AOVType::COUNT)];

__device__ AOV & get_aov(AOVType aov_type) {
    return aovs[size_t(aov_type)];
}

__device__ inline float4 aov_framebuffer_get(AOVType aov_type, int pixel_index) {
    const AOV & aov = get_aov(aov_type);
    assert(aov.framebuffer);
    return aov.framebuffer[pixel_index];
}

__device__ inline void aov_framebuffer_set(AOVType aov_type, int pixel_index, float4 value) {
    AOV & aov = get_aov(aov_type);
    if (aov.framebuffer) {
        aov.framebuffer[pixel_index] = value;
    }
}

__device__ inline void aov_framebuffer_add(AOVType aov_type, int pixel_index, float4 value) {
    AOV & aov = get_aov(aov_type);
    if (aov.framebuffer) {
        aov.framebuffer[pixel_index] += value;
    }
}

__device__ inline float4 aov_accumulate(AOVType aov_type, int pixel_index, float n) {
    AOV & aov = get_aov(aov_type);
    if (aov.framebuffer) {
        if (n > 0.0f) {
            aov.accumulator[pixel_index] += (aov.framebuffer[pixel_index] - aov.accumulator[pixel_index]) / n; // Online average
        } else {
            aov.accumulator[pixel_index] = aov.framebuffer[pixel_index];
        }
        return aov.accumulator[pixel_index];
    }
    return make_float4(0.0f);
}
