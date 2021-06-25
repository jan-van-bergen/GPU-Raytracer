#pragma once

// Glossy materials with roughness below the cutoff don't use direct Light sampling
#define ROUGHNESS_CUTOFF 0.3f

__device__ const Texture<float4> * textures;

__device__ inline float2 texture_get_size(int texture_id) {
	if (texture_id == INVALID) {
		return make_float2(0.0f, 0.0f);
	} else {
		return textures[texture_id].size;
	}
}

enum struct MaterialType : char {
	LIGHT      = 0,
	DIFFUSE    = 1,
	DIELECTRIC = 2,
	GLOSSY     = 3
};

struct Material {
	union {
		struct {
			float4 emission;
		} light;
		struct {
			float4 diffuse_and_texture_id;
		} diffuse;
		struct {
			float4 negative_absorption_and_ior;
		} dielectric;
		struct {
			float4 diffuse_and_texture_id;
			float2 ior_and_roughness;
		} glossy;
	};
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

struct MaterialDielectric {
	float3 negative_absorption;
	float  index_of_refraction;
};

struct MaterialGlossy {
	float3 diffuse;
	int    texture_id;
	float  index_of_refraction;
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

__device__ inline MaterialDielectric material_as_dielectric(int material_id) {
	float4 negative_absorption_and_ior = __ldg(&materials[material_id].dielectric.negative_absorption_and_ior);

	MaterialDielectric material;
	material.negative_absorption = make_float3(negative_absorption_and_ior);
	material.index_of_refraction = negative_absorption_and_ior.w;
	return material;
}

__device__ inline MaterialGlossy material_as_glossy(int material_id) {
	float4 diffuse_and_texture_id = __ldg(&materials[material_id].glossy.diffuse_and_texture_id);
	float2 ior_and_roughness      = __ldg(&materials[material_id].glossy.ior_and_roughness);

	MaterialGlossy material;
	material.diffuse             = make_float3(diffuse_and_texture_id);
	material.texture_id          = __float_as_int(diffuse_and_texture_id.w);
	material.index_of_refraction = ior_and_roughness.x;
	material.roughness           = ior_and_roughness.y;
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

__device__ __constant__ float lights_total_power;

__device__ __constant__ const int   * light_indices;
__device__ __constant__ const float * light_power_cumulative;

__device__ __constant__ const int   * light_mesh_triangle_count;
__device__ __constant__ const int   * light_mesh_triangle_first_index;
__device__ __constant__ const float * light_mesh_power_scaled;
__device__ __constant__ const float * light_mesh_power_unscaled;
__device__ __constant__ const int   * light_mesh_transform_indices;

// Assumes no Total Internal Reflection
__device__ inline float fresnel(float eta_1, float eta_2, float cos_theta_i, float cos_theta_t) {
	float s = (eta_1 * cos_theta_i - eta_2 * cos_theta_t) / (eta_1 * cos_theta_i + eta_2 * cos_theta_t);
	float p = (eta_1 * cos_theta_t - eta_2 * cos_theta_i) / (eta_1 * cos_theta_t + eta_2 * cos_theta_i);

	return 0.5f * (s*s + p*p);
}

__device__ inline float fresnel_schlick(float eta_1, float eta_2, float cos_theta_i) {
	float r_0 = (eta_1 - eta_2) / (eta_1 + eta_2);
	r_0 *= r_0;

	return r_0 + (1.0f - r_0) * (cos_theta_i * cos_theta_i * cos_theta_i * cos_theta_i * cos_theta_i);
}

// Distribution of Normals term D for the Beckmann microfacet model
__device__ inline float beckmann_D(float m_dot_n, float alpha) {
	if (m_dot_n <= 0.0f) return 0.0f;

	float cos_theta_m  = m_dot_n;
	float cos2_theta_m = cos_theta_m  * cos_theta_m;
	float cos4_theta_m = cos2_theta_m * cos2_theta_m;

	float tan2_theta_m = max(0.0f, 1.0f - cos2_theta_m) / cos2_theta_m; // tan^2(x) = sec^2(x) - 1 = (1 - cos^2(x)) / cos^2(x)

	float alpha2 = alpha * alpha;

	return exp(-tan2_theta_m / alpha2) / (PI * alpha2 * cos4_theta_m);
}

// Monodirectional Smith shadowing term G1 for the Beckmann microfacet model
__device__ inline float beckmann_G1(float v_dot_n, float v_dot_m, float alpha) {
	if (v_dot_m <= 0.0f || v_dot_n <= 0.0f) return 0.0f;

	float cos_theta_v = v_dot_n;
	float tan_theta_v = sqrt(max(1e-8f, 1.0f - cos_theta_v*cos_theta_v)) / cos_theta_v; // tan(acos(x)) = sqrt(1 - x^2) / x

	float a = 1.0f / (alpha * tan_theta_v);
	if (a >= 1.6f) return 1.0f;

	// Rational approximation
	return (a * (3.535f + 2.181f * a)) / (1.0f + a * (2.276f + 2.577f * a));
}

// Rational approximation to lambda term for the Beckmann microfacet model
__device__ float beckmann_lambda(float cos_theta, float alpha) {
	float tan_theta_v = sqrtf(max(1e-8f, 1.0f - cos_theta*cos_theta)) / cos_theta;
	float a = 1.0f / (alpha * tan_theta_v);

	if (a >= 1.6f) return 0.0f;

	return (1.0f - a * (1.259f + 0.396f*a)) / (a * (3.535f + 2.181f*a));
};

// Distribution of Normals term D for the GGX microfacet model
__device__ inline float ggx_D(float m_dot_n, float alpha) {
	if (m_dot_n <= 0.0f) return 0.0f;

	float cos_theta  = m_dot_n;
	float cos_theta2 = cos_theta  * cos_theta;
	float cos_theta4 = cos_theta2 * cos_theta2;

	float tan_theta2 = (1.0f - cos_theta2) / cos_theta2;

	float alpha2 = alpha * alpha;
	float alpha2_plus_tan_theta2 = alpha2 + tan_theta2;

	return alpha2 / (PI * cos_theta4 * alpha2_plus_tan_theta2 * alpha2_plus_tan_theta2);
}

// Monodirectional Smith shadowing term G1 for the GGX microfacet model
__device__ inline float ggx_G1(float v_dot_n, float v_dot_m, float alpha) {
	if (v_dot_m <= 0.0f || v_dot_n <= 0.0f) return 0.0f;

	float cos_theta  = v_dot_n;
	float cos_theta2 = cos_theta  * cos_theta;

	float tan_theta2 = (1.0f - cos_theta2) / cos_theta2;

	float alpha2 = alpha * alpha;

	return 2.0f / (1.0f - sqrtf(1.0f + alpha2 * tan_theta2));
}

// Lambda term for the GGX microfacet model
__device__ inline float ggx_lambda(float cos_theta, float alpha) {
	float cos_theta2 = cos_theta  * cos_theta;
	float tan_theta2 = (1.0f - cos_theta2) / cos_theta2;

	float one_over_a2 = alpha*alpha * tan_theta2;

	return (sqrtf(1.0f + one_over_a2) - 1.0f) * 0.5f;
}

__device__ inline float microfacet_D(float m_dot_n, float alpha) {
#if MICROFACET_MODEL == MICROFACET_BECKMANN
	return beckmann_D(m_dot_n, alpha);
#elif MICROFACET_MODEL == MICROFACET_GGX
	return ggx_D(m_dot_n, alpha);
#endif
}

// Shadowing/Masking term for the given microfacet model
// If MICROFACET_SEPARATE_G_TERMS is set to true, two separate monodirectional Smith terms G1 are used,
// otherwise a Height-Correlated Masking and Shadowing term G2 is used based on 2 lambda terms.
__device__ inline float microfacet_G(float i_dot_m, float o_dot_m, float i_dot_n, float o_dot_n, float m_dot_n, float alpha) {
#if MICROFACET_SEPARATE_G_TERMS
	#if MICROFACET_MODEL == MICROFACET_BECKMANN
		return
			beckmann_G1(i_dot_n, m_dot_n, alpha) * 
			beckmann_G1(o_dot_n, m_dot_n, alpha); 
	#elif MICROFACET_MODEL == MICROFACET_GGX
		return
			ggx_G1(i_dot_n, m_dot_n, alpha) * 
			ggx_G1(o_dot_n, m_dot_n, alpha); 
	#endif
#else
	#if MICROFACET_MODEL == MICROFACET_BECKMANN
		const auto lambda = beckmann_lambda;
	#elif MICROFACET_MODEL == MICROFACET_GGX
		const auto lambda = ggx_lambda;
	#endif

	if (i_dot_m <= 0.0f || o_dot_m <= 0.0f) return 0.0f;
	
	return 1.0f / (1.0f + lambda(i_dot_n, alpha) + lambda(o_dot_m, alpha));
#endif
}
