#pragma once

// Glossy materials with roughness below the cutoff don't use direct Light sampling
#define ROUGHNESS_CUTOFF (0.1f * 0.1f)

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

// Distribution of Normals term D for the GGX microfacet model
__device__ inline float ggx_D(const float3 & micro_normal, float alpha_x, float alpha_y) {
	float sx = -micro_normal.x / (micro_normal.z * alpha_x);
	float sy = -micro_normal.y / (micro_normal.z * alpha_y);

	float sl = 1.0f + sx * sx + sy * sy;

	float cos_theta_2 = micro_normal.z * micro_normal.z;
	float cos_theta_4 = cos_theta_2 * cos_theta_2;

	return 1.0f / (sl * sl * PI * alpha_x * alpha_y * cos_theta_4);
}

// Monodirectional Smith shadowing/masking term G1 for the GGX microfacet model
__device__ inline float ggx_G1(const float3 & omega, float alpha_x2, float alpha_y2) {
	float cos_o2 = omega.z * omega.z;
	float tan_theta_o2 = (1.0f - cos_o2) / cos_o2;
	float cos_phi_o2 = omega.x * omega.x;
	float sin_phi_o2 = omega.y * omega.y;

	float alpha_o2 = (cos_phi_o2 * alpha_x2 + sin_phi_o2 * alpha_y2) / (cos_phi_o2 + sin_phi_o2);

	return 2.0f / (1.0f + sqrtf(fmaxf(0.0f, 1.0f + alpha_o2 * tan_theta_o2)));
}

// Based on: Sampling the GGX Distribution of Visible Normals - Heitz 2018
__device__ inline float3 ggx_sample_distribution_of_normals(const float3 & omega, float alpha_x, float alpha_y, float u1, float u2){
	// Transform the view direction to the hemisphere configuration
	float3 v = normalize(make_float3(alpha_x * omega.x, alpha_y * omega.y, omega.z));

	// Orthonormal basis (with special case if cross product is zero)
	float length_squared = v.x*v.x + v.y*v.y;
	float3 axis_1 = length_squared > 0.0f ? make_float3(-v.y, v.x, 0.0f) / sqrt(length_squared) : make_float3(1.0f, 0.0f, 0.0f);
	float3 axis_2 = cross(v, axis_1);

	// Parameterization of the projected area
	float r = sqrt(u1);
	float phi = TWO_PI * u2;

	float sin_phi;
	float cos_phi;
	__sincosf(phi, &sin_phi, &cos_phi);

	float t1 = r * cos_phi;
	float t2 = r * sin_phi;

	float s = 0.5f * (1.0f + v.z);
	t2 = (1.0f - s) * sqrtf(1.0 - t1*t1) + s*t2;

	// Reproject onto hemisphere
	float3 n_h = t1*axis_1 + t2*axis_2 + sqrtf(fmaxf(0.0f, 1.0f - t1*t1 - t2*t2)) * v;

	// Transform the normal back to the ellipsoid configuration
	return normalize(make_float3(alpha_x * n_h.x, alpha_y * n_h.y, fmaxf(0.0f, n_h.z)));
}

__device__ inline float ggx_eval(const float3 & omega_o, const float3 & omega_i, float ior, float alpha_x, float alpha_y, float & pdf) {
	float3 half_vector = normalize(omega_o + omega_i);
	float mu = fmaxf(0.0, dot(omega_o, half_vector));

	float F = fresnel_schlick(ior, 1.0f, mu);
	float D = ggx_D(half_vector, alpha_x, alpha_y);

	// Masking/shadowing using two monodirectional Smith terms
	float G1_o = ggx_G1(omega_o, alpha_x * alpha_x, alpha_y * alpha_y);
	float G1_i = ggx_G1(omega_i, alpha_x * alpha_x, alpha_y * alpha_y);
	float G2 = G1_o * G1_i;

	float denom = 4.0f * omega_i.z * omega_o.z;

	pdf = G1_o * D * mu / denom;
	return F * D * G2 / denom;
}
