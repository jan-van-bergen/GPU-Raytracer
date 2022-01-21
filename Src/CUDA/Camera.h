#pragma once
#include "Sampling.h"

constexpr int TAA_HALTON_NUM_SAMPLES = 4;

__device__ __constant__ const float taa_halton_x[TAA_HALTON_NUM_SAMPLES] = { 0.3f, 0.7f, 0.2f, 0.8f };
__device__ __constant__ const float taa_halton_y[TAA_HALTON_NUM_SAMPLES] = { 0.2f, 0.8f, 0.7f, 0.3f };

struct Camera {
	float3 position;
	float3 bottom_left_corner;
	float3 x_axis;
	float3 y_axis;
	float  pixel_spread_angle;
	float  aperture_radius;
	float  focal_distance;
};

__device__ inline Ray camera_generate_ray(int pixel_index, int sample_index, int x, int y, const Camera & camera) {
	float2 rand_filter   = random<SampleDimension::FILTER>  (pixel_index, 0, sample_index);
	float2 rand_aperture = random<SampleDimension::APERTURE>(pixel_index, 0, sample_index);

	float2 jitter;

	if (config.enable_svgf) {
		jitter.x = taa_halton_x[sample_index & (TAA_HALTON_NUM_SAMPLES-1)];
		jitter.y = taa_halton_y[sample_index & (TAA_HALTON_NUM_SAMPLES-1)];
	} else {
		switch (config.reconstruction_filter) {
			case ReconstructionFilter::BOX: {
				jitter = rand_filter;
				break;
			}
			case ReconstructionFilter::TENT: {
				jitter.x = sample_tent(rand_filter.x);
				jitter.y = sample_tent(rand_filter.y);
				break;
			}
			case ReconstructionFilter::GAUSSIAN: {
				float2 gaussians = sample_gaussian(rand_filter.x, rand_filter.y);
				jitter.x = 0.5f + 0.5f * gaussians.x;
				jitter.y = 0.5f + 0.5f * gaussians.y;
				break;
			}
		}
	}

	float x_jittered = float(x) + jitter.x;
	float y_jittered = float(y) + jitter.y;

	float3 focal_point = camera.focal_distance * normalize(camera.bottom_left_corner + x_jittered * camera.x_axis + y_jittered * camera.y_axis);
	float2 lens_point  = camera.aperture_radius * sample_disk(rand_aperture.x, rand_aperture.y);

	float3 offset = camera.x_axis * lens_point.x + camera.y_axis * lens_point.y;
	float3 direction = normalize(focal_point - offset);

    Ray ray;
    ray.origin = camera.position + offset;
    ray.direction = direction;
    return ray;
}
