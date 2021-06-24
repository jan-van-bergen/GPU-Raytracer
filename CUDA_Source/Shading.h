#pragma once

// Glossy materials with roughness below the cutoff don't use direct Light sampling
#define ROUGHNESS_CUTOFF 0.3f

__device__ const Texture<float4> * textures;

struct Material {
	enum class Type : char {
		LIGHT      = 0,
		DIFFUSE    = 1,
		DIELECTRIC = 2,
		GLOSSY     = 3
	};

	Type type;

	float3 diffuse;
	int texture_id;

	float3 emission;

	float index_of_refraction;
	float3 negative_absorption;

	float roughness;

	__device__ inline float3 albedo(float s, float t) const {
		if (texture_id == INVALID) return diffuse;

		float4 tex_colour = textures[texture_id].get(s, t);
		return diffuse * make_float3(tex_colour);
	}

	__device__ inline float3 albedo(float s, float t, float lod) const {
		if (texture_id == INVALID) return diffuse;

		float4 tex_colour = textures[texture_id].get_lod(s, t, lod);
		return diffuse * make_float3(tex_colour);
	}

	__device__ inline float3 albedo(float s, float t, float2 dx, float2 dy) const {
		if (texture_id == INVALID) return diffuse;

		float4 tex_colour = textures[texture_id].get_grad(s, t, dx, dy);
		return diffuse * make_float3(tex_colour);
	}

	__device__ inline float2 get_texture_size() const {
		if (texture_id == INVALID) {
			return make_float2(0.0f, 0.0f);
		} else {
			return textures[texture_id].size;
		}
	}
};

__device__ __constant__ const Material * materials;

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
