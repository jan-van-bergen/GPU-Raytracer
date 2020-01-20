#pragma once
#include <vector_types.h>

#include "cuda_math.h"

#include "../Common.h"

struct Material;

__device__ Material            * materials;
__device__ cudaTextureObject_t * textures;

struct Material {
	enum Type : char{
		DIFFUSE    = 0,
		DIELECTRIC = 1,
		GLOSSY     = 2
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

	__device__ bool is_light() const {
		return dot(emittance, emittance) > 0.0f;
	}
};

__device__ int     light_count;
__device__ int   * light_indices;
__device__ float * light_areas;
__device__ float total_light_area;

// Monodirectional shadowing term G1 for the Smith shadowing function G for the Beckmann distribution
__device__ float beckmann_g1(const float3 & v, const float3 & m, const float3 & n, float alpha) {
	float v_dot_n = dot(v, n);
	if (dot(v, m) / v_dot_n <= 0.0f) return 0.0f;

	float tan_theta_v = sqrt(1.0f - v_dot_n*v_dot_n) / v_dot_n; // tan(acos(x)) = sqrt(1 - x^2) / x
	float one_over_a  = alpha * tan_theta_v;
	
	// Check if a >= 1.6 by checking if 1/a <= 1/1.6
	if (one_over_a <= 0.625f) return 1.0f;

	// Rational approximation
	float a = 1.0f / one_over_a;
	return (3.535f * a + 2.181f * a*a) / (1.0f + 2.276f * a + 2.577f * a*a);
}
