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
