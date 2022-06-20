#pragma once

struct Medium {
	float4 sigma_a_and_g;
	float4 sigma_s;
};

__device__ __constant__ Medium * media;

struct HomogeneousMedium {
	float3 sigma_a;
	float3 sigma_s;
	float  g;
};

__device__ HomogeneousMedium medium_as_homogeneous(int medium_id) {
	float4 sigma_a_and_g = __ldg(&media[medium_id].sigma_a_and_g);
	float4 sigma_s       = __ldg(&media[medium_id].sigma_s);

	HomogeneousMedium medium = { };
	medium.sigma_a = make_float3(sigma_a_and_g);
	medium.sigma_s = make_float3(sigma_s);
	medium.g       = sigma_a_and_g.w;
	return medium;
}

__device__ inline float3 beer_lambert(const float3 & sigma_t, float distance) {
	return make_float3(
		expf(-sigma_t.x * distance),
		expf(-sigma_t.y * distance),
		expf(-sigma_t.z * distance)
	);
}
