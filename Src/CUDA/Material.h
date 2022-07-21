#pragma once
#include "Sampling.h"

// Microfacet materials with roughness below the cutoff don't use direct Light sampling
#define ROUGHNESS_CUTOFF (0.05f)

__device__ inline float roughness_to_alpha(float linear_roughness) {
	return fmaxf(1e-6f, square(linear_roughness));
}

__device__ const Texture<float4> * textures;

enum struct MaterialType : char {
	LIGHT,
	DIFFUSE,
	PLASTIC,
	DIELECTRIC,
	CONDUCTOR
};

union Material {
	struct {
		float4 emission;
	} light;
	struct {
		float4 diffuse_and_texture_id;
	} diffuse;
	struct {
		float4 diffuse_and_texture_id;
		float  linear_roughness;
	} plastic;
	struct {
		float4 medium_ior_and_linear_roughness;
	} dielectric;
	struct {
		float4 eta_and_linear_roughness;
		float4 k;
	} conductor;
};

__device__ __constant__ const MaterialType * material_types;
__device__ __constant__ const Material     * materials;

__device__ inline MaterialType material_get_type(int material_id) {
	return material_types[material_id];
}

struct MaterialLight {
	float3 emission;
};

struct MaterialDiffuse {
	float3 diffuse;
	int    texture_id;
};

struct MaterialPlastic {
	float3 diffuse;
	int    texture_id;
	float  linear_roughness;
};

struct MaterialDielectric {
	int   medium_id;
	float ior;
	float linear_roughness;
};

struct MaterialConductor {
	float3 eta;
	float  linear_roughness;
	float3 k;
};

__device__ inline MaterialLight material_as_light(int material_id) {
	float4 emission = __ldg(&materials[material_id].light.emission);

	MaterialLight material;
	material.emission = make_float3(emission);
	return material;
}

__device__ inline MaterialDiffuse material_as_diffuse(int material_id) {
	float4 diffuse_and_texture_id = __ldg(&materials[material_id].diffuse.diffuse_and_texture_id);
	
	MaterialDiffuse material;
	material.diffuse    = make_float3(diffuse_and_texture_id);
	material.texture_id = __float_as_int(diffuse_and_texture_id.w);
	return material;
}

__device__ inline MaterialPlastic material_as_plastic(int material_id) {
	float4 diffuse_and_texture_id = __ldg(&materials[material_id].plastic.diffuse_and_texture_id);
	float  linear_roughness       = __ldg(&materials[material_id].plastic.linear_roughness);

	MaterialPlastic material;
	material.diffuse          = make_float3(diffuse_and_texture_id);
	material.texture_id       = __float_as_int(diffuse_and_texture_id.w);
	material.linear_roughness = linear_roughness;
	return material;
}

__device__ inline MaterialDielectric material_as_dielectric(int material_id) {
	float4 medium_ior_and_linear_roughness = __ldg(&materials[material_id].dielectric.medium_ior_and_linear_roughness);

	MaterialDielectric material;
	material.medium_id        = __float_as_int(medium_ior_and_linear_roughness.x);
	material.ior              = medium_ior_and_linear_roughness.y;
	material.linear_roughness = medium_ior_and_linear_roughness.z;
	return material;
}

__device__ inline MaterialConductor material_as_conductor(int material_id) {
	float4 eta_and_linear_roughness = __ldg(&materials[material_id].conductor.eta_and_linear_roughness);
	float4 k                        = __ldg(&materials[material_id].conductor.k);

	MaterialConductor material;
	material.eta              = make_float3(eta_and_linear_roughness);
	material.linear_roughness = eta_and_linear_roughness.w;
	material.k                = make_float3(k);
	return material;
}

__device__ inline float3 material_get_albedo(float3 diffuse, int texture_id, float s, float t) {
	if (texture_id == INVALID) return diffuse;

	float4 tex_colour = textures[texture_id].get(s, t);
	return diffuse * make_float3(tex_colour);
}

__device__ inline float3 material_get_albedo(float3 diffuse, int texture_id, float s, float t, float lod) {
	if (texture_id == INVALID) return diffuse;

	float4 tex_colour = textures[texture_id].get_lod(s, t, lod);
	return diffuse * make_float3(tex_colour);
}

__device__ inline float3 material_get_albedo(float3 diffuse, int texture_id, float s, float t, float2 dx, float2 dy) {
	if (texture_id == INVALID) return diffuse;

	float4 tex_colour = textures[texture_id].get_grad(s, t, dx, dy);
	return diffuse * make_float3(tex_colour);
}

__device__ inline float fresnel_dielectric(float cos_theta_i, float eta) {
	float sin_theta_o2 = eta*eta * (1.0f - square(cos_theta_i));
	if (sin_theta_o2 >= 1.0f) {
		return 1.0f; // Total internal reflection (TIR)
	}

	float cos_theta_o = safe_sqrt(1.0f - sin_theta_o2);

	float p = divide_difference_by_sum(eta * cos_theta_i, cos_theta_o);
	float s = divide_difference_by_sum(cos_theta_i, eta * cos_theta_o);

	return 0.5f * (p*p + s*s);
}

// Formula from Shirley - Physically Based Lighting Calculations for Computer Graphics
__device__ inline float3 fresnel_conductor(float cos_theta_i, float3 eta, float3 k) {
	float cos_theta_i2 = square(cos_theta_i);
	float sin_theta_i2 = 1.0f - cos_theta_i2;

	float3 inner      = eta*eta - k*k - sin_theta_i2;
	float3 a2_plus_b2 = safe_sqrt(inner*inner + 4.0f * k*k * eta*eta);
	float3 a          = safe_sqrt(0.5f * (a2_plus_b2 + inner));

	float3 s2 = divide_difference_by_sum(a2_plus_b2 + cos_theta_i2,                        2.0f * a * cos_theta_i);
	float3 p2 = divide_difference_by_sum(a2_plus_b2 * cos_theta_i2 + square(sin_theta_i2), 2.0f * a * cos_theta_i * sin_theta_i2) * s2;

	return 0.5f * (p2 + s2);
}

__device__ inline float average_fresnel(float ior) {
	// Approximation by Kully-Conta 2017
	return (ior - 1.0f) / (4.08567f + 1.00071f*ior);
}

__device__ inline float3 average_fresnel(float3 eta, float3 k) {
	// Approximation by d'Eon (Hitchikers Guide to Multiple Scattering)
	float3 numerator   = eta*(133.736f - 98.9833f*eta) + k*(eta*(59.5617f - 3.98288f*eta) - 182.37f) + ((0.30818f*eta - 13.1093f)*eta - 62.5919f)*k*k - 8.21474f;
	float3 denominator = k*(eta*(94.6517f - 15.8558f*eta) - 187.166f) + (-78.476*eta - 395.268f)*eta + (eta*(eta - 15.4387f) - 62.0752f)*k*k;
	return numerator / denominator;
}

// Distribution of Normals term D for the GGX microfacet model
__device__ inline float ggx_D(float3 micro_normal, float alpha_x, float alpha_y) {
	if (micro_normal.z < 1e-6f) {
		return 0.0f;
	}

	float sx = -micro_normal.x / (micro_normal.z * alpha_x);
	float sy = -micro_normal.y / (micro_normal.z * alpha_y);

	float sl = 1.0f + sx * sx + sy * sy;

	float cos_theta_2 = micro_normal.z * micro_normal.z;
	float cos_theta_4 = cos_theta_2 * cos_theta_2;

	return 1.0f / (sl * sl * PI * alpha_x * alpha_y * cos_theta_4);
}

__device__ inline float ggx_lambda(float3 omega, float alpha_x, float alpha_y) {
	return 0.5f * (sqrtf(1.0f + (square(alpha_x * omega.x) + square(alpha_y * omega.y)) / square(omega.z)) - 1.0f);
}

// Monodirectional Smith shadowing/masking term
__device__ inline float ggx_G1(float3 omega, float alpha_x, float alpha_y) {
	return 1.0f / (1.0f + ggx_lambda(omega, alpha_x, alpha_y));
}

// Height correlated shadowing and masking term
__device__ inline float ggx_G2(float3 omega_o, float3 omega_i, float3 omega_m, float alpha_x, float alpha_y) {
	bool omega_i_backfacing = dot(omega_i, omega_m) * omega_i.z <= 0.0f;
	bool omega_o_backfacing = dot(omega_o, omega_m) * omega_o.z <= 0.0f;

	if (omega_i_backfacing || omega_o_backfacing) {
		return 0.0f;
	} else {
		return 1.0f / (1.0f + ggx_lambda(omega_o, alpha_x, alpha_y) + ggx_lambda(omega_i, alpha_x, alpha_y));
	}
}
