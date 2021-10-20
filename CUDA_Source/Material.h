#pragma once
#include "Sampling.h"

// Microfacet materials with roughness below the cutoff don't use direct Light sampling
#define ROUGHNESS_CUTOFF (0.001f)

__device__ const Texture<float4> * textures;

__device__ inline float2 texture_get_size(int texture_id) {
	if (texture_id == INVALID) {
		return make_float2(0.0f, 0.0f);
	} else {
		return textures[texture_id].size;
	}
}

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
		float  roughness;
	} plastic;
	struct {
		float4 negative_absorption_and_ior;
		float  roughness;
	} dielectric;
	struct {
		float4 diffuse_and_texture_id;
		float4 eta_and_k;       // eta xyz and k x
		float4 k_and_roughness; // k yz and roughness;
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
	float  roughness;
};

struct MaterialDielectric {
	float3 negative_absorption;
	float  index_of_refraction;
	float  roughness;
};

struct MaterialConductor {
	float3 diffuse;
	int    texture_id;
	float3 eta;
	float3 k;
	float  roughness;
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
	float  roughness              = __ldg(&materials[material_id].plastic.roughness);

	MaterialPlastic material;
	material.diffuse    = make_float3(diffuse_and_texture_id);
	material.texture_id = __float_as_int(diffuse_and_texture_id.w);
	material.roughness  = roughness;
	return material;
}

__device__ inline MaterialDielectric material_as_dielectric(int material_id) {
	float4 negative_absorption_and_ior = __ldg(&materials[material_id].dielectric.negative_absorption_and_ior);
	float  roughness                   = __ldg(&materials[material_id].dielectric.roughness);

	MaterialDielectric material;
	material.negative_absorption = make_float3(negative_absorption_and_ior);
	material.index_of_refraction = negative_absorption_and_ior.w;
	material.roughness           = roughness;
	return material;
}

__device__ inline MaterialConductor material_as_conductor(int material_id) {
	float4 diffuse_and_texture_id = __ldg(&materials[material_id].conductor.diffuse_and_texture_id);
	float4 eta_and_k              = __ldg(&materials[material_id].conductor.eta_and_k);
	float4 k_and_roughness        = __ldg(&materials[material_id].conductor.k_and_roughness);

	MaterialConductor material;
	material.diffuse             = make_float3(diffuse_and_texture_id);
	material.texture_id          = __float_as_int(diffuse_and_texture_id.w);
	material.eta                 = make_float3(eta_and_k);
	material.k                   = make_float3(eta_and_k.w, k_and_roughness.x, k_and_roughness.y);
	material.roughness           = k_and_roughness.z;
	return material;
}

__device__ inline float3 material_get_albedo(const float3 & diffuse, int texture_id, float s, float t) {
	if (texture_id == INVALID) return diffuse;

	float4 tex_colour = textures[texture_id].get(s, t);
	return diffuse * make_float3(tex_colour);
}

__device__ inline float3 material_get_albedo(const float3 & diffuse, int texture_id, float s, float t, float lod) {
	if (texture_id == INVALID) return diffuse;

	float4 tex_colour = textures[texture_id].get_lod(s, t, lod);
	return diffuse * make_float3(tex_colour);
}

__device__ inline float3 material_get_albedo(const float3 & diffuse, int texture_id, float s, float t, float2 dx, float2 dy) {
	if (texture_id == INVALID) return diffuse;

	float4 tex_colour = textures[texture_id].get_grad(s, t, dx, dy);
	return diffuse * make_float3(tex_colour);
}

__device__ inline float fresnel_dielectric(float cos_theta_i, float eta) {
	float sin_theta_o2 = eta*eta * (1.0f - cos_theta_i*cos_theta_i);
	if (sin_theta_o2 >= 1.0f) {
		return 1.0f; // Total internal reflection (TIR)
	}

	float cos_theta_o = sqrtf(1.0f - sin_theta_o2);

	float s = (cos_theta_i - eta * cos_theta_o) / (cos_theta_i + eta * cos_theta_o);
	float p = (eta * cos_theta_i - cos_theta_o) / (eta * cos_theta_i + cos_theta_o);

	return 0.5f * (p*p + s*s);
}

__device__ inline float3 fresnel_conductor(float cos_theta_i, const float3 & eta, const float3 & k) {
	float cos_theta_i2 = cos_theta_i * cos_theta_i;

	float3 t1 = eta*eta + k*k;
	float3 t0 = t1 * cos_theta_i;

	float3 p2 = (t0 - (eta * (2.0f * cos_theta_i)) + make_float3(1.0f))         / (t0 + (eta * (2.0f * cos_theta_i)) + make_float3(1.0f));
	float3 s2 = (t1 - (eta * (2.0f * cos_theta_i)) + make_float3(cos_theta_i2)) / (t1 + (eta * (2.0f * cos_theta_i)) + make_float3(cos_theta_i2));

	return 0.5f * (p2 + s2);
}

// Distribution of Normals term D for the GGX microfacet model
__device__ inline float ggx_D(const float3 & micro_normal, float alpha_x, float alpha_y) {
	float sx = -micro_normal.x / (micro_normal.z * alpha_x);
	float sy = -micro_normal.y / (micro_normal.z * alpha_y);

	float sl = 1.0f + sx * sx + sy * sy;

	float cos_theta_2 = micro_normal.z * micro_normal.z;
	float cos_theta_4 = cos_theta_2 * cos_theta_2;

	return 1.0f / (sl * sl * PI * alpha_x * alpha_y * cos_theta_4);
}

__device__ inline float ggx_lambda(const float3 & omega, float alpha_x, float alpha_y) {
	return 0.5f * (sqrtf(1.0f + (square(alpha_x * omega.x) + square(alpha_y * omega.y)) / square(omega.z)) - 1.0f);
}

// Monodirectional Smith shadowing/masking term
__device__ inline float ggx_G1(const float3 & omega, float alpha_x, float alpha_y) {
	return 1.0f / (1.0f + ggx_lambda(omega, alpha_x, alpha_y));
}

// Height correlated shadowing and masking term
__device__ inline float ggx_G2(const float3 & omega_o, const float3 & omega_i, const float3 & omega_m, float alpha_x, float alpha_y) {
	bool omega_i_backfacing = dot(omega_i, omega_m) * omega_i.z <= 0.0f;
	bool omega_o_backfacing = dot(omega_o, omega_m) * omega_o.z <= 0.0f;

	if (omega_i_backfacing || omega_o_backfacing) {
		return 0.0f;
	} else {
		return 1.0f / (1.0f + ggx_lambda(omega_o, alpha_x, alpha_y) + ggx_lambda(omega_i, alpha_x, alpha_y));
	}
}

__device__ inline float3 ggx_eval(const MaterialConductor & material, const float3 & omega_o, const float3 & omega_i, float & pdf) {
	float alpha_x = material.roughness;
	float alpha_y = material.roughness; // TODO: anisotropic
	
	float3 omega_m = normalize(omega_o + omega_i);
	float mu = fmaxf(0.0, dot(omega_o, omega_m));

	float3 F  = fresnel_conductor(mu, material.eta, material.k);
	float  D  = ggx_D (omega_m, alpha_x, alpha_y);
	float  G1 = ggx_G1(omega_i, alpha_x, alpha_y);
	float  G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

	pdf = G1 * D / (4.0f * omega_i.z);
	return F * D * G2 / (4.0f * omega_i.z); // BRDF * cos(theta_o)
}

__device__ inline float3 ggx_sample(const MaterialConductor & material, float u1, float u2, const float3 & omega_i, float3 & omega_o, float & pdf) {
	float alpha_x = material.roughness;
	float alpha_y = material.roughness; // TODO: anisotropic
	
	float3 omega_m = sample_visible_normals_ggx(omega_i, material.roughness, material.roughness, u1, u2);
	omega_o = reflect(-omega_i, omega_m);

	float mu = fmaxf(0.0, dot(omega_o, omega_m));

	float3 F  = fresnel_conductor(mu, material.eta, material.k);
	float  D  = ggx_D (omega_m, alpha_x, alpha_y);
	float  G1 = ggx_G1(omega_i, alpha_x, alpha_y);
	float  G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

	pdf = G1 * D / (4.0f * omega_i.z);
	return F * G2 / G1; // BRDF * cos(theta_o) / pdf
}
