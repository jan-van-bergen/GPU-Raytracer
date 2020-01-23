#pragma once
#include <vector_types.h>

#include "cuda_math.h"

#include "../Common.h"
#include "Util.h"

// Glossy materials with roughness below the cutoff don't use direct Light sampling
#define ROUGNESS_CUTOFF 0.1f

struct Material;

__device__ Material            * materials;
__device__ cudaTextureObject_t * textures;

struct Material {
	enum Type : char{
		LIGHT      = 0,
		DIFFUSE    = 1,
		DIELECTRIC = 2,
		GLOSSY     = 3
	};

	Type type;

	float3 diffuse;
	int texture_id;

	float3 emittance;

	float index_of_refraction;

	float roughness;

	__device__ float3 albedo(float u, float v) const {
		if (texture_id == -1) return diffuse;

		float4 tex_colour;

		for (int i = 0; i < MAX_TEXTURES; i++) {
			if (texture_id == i) {
				tex_colour = tex2D<float4>(textures[i], u, v);
			}
		}

		return diffuse * make_float3(tex_colour);
	}
};

__device__ int     light_count;
__device__ int   * light_indices;
__device__ float * light_areas;
__device__ float total_light_area;

__device__ float beckmann_D(float m_dot_n, float alpha) {
	if (m_dot_n <= 0.0f) return 0.0f;

	float cos_theta_m  = m_dot_n;
	float cos2_theta_m = cos_theta_m  * cos_theta_m;
	float cos4_theta_m = cos2_theta_m * cos2_theta_m;

	float tan2_theta_m  = max(0.0f, 1.0f - cos2_theta_m) / cos2_theta_m; // tan^2(x) = sec^2(x) - 1 = (1 - cos^2(x)) / cos^2(x)

	float alpha2 = alpha * alpha;

 	return exp(-tan2_theta_m / alpha2) / (PI * alpha2 * cos4_theta_m);
}

// Monodirectional shadowing term G1 for the Smith shadowing function G for the Beckmann distribution
__device__ float beckmann_G1(float v_dot_n, float v_dot_m, float alpha) {
	if (v_dot_m / v_dot_n <= 0.0f) return 0.0f;
	float cos_theta_v = v_dot_n;

	float tan_theta_v = sqrt(max(1e-8f, 1.0f - cos_theta_v*cos_theta_v)) / cos_theta_v; // tan(acos(x)) = sqrt(1 - x^2) / x
	float one_over_a  = alpha * tan_theta_v;
	
	// Check if a >= 1.6 by checking if 1/a <= 1/1.6
	if (one_over_a <= 0.625f) return 1.0f;

	// Rational approximation
	float a = 1.0f / one_over_a;
	return (a * (3.535f + 2.181f * a)) / (1.0f + a * (2.276f + 2.577f * a));
}
