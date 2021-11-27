#pragma once
#include "Sampling.h"
#include "Material.h"
#include "RayCone.h"

struct BSDFDiffuse {
	static constexpr bool HAS_ALBEDO = true;

	int pixel_index;
	int bounce;
	int sample_index;

	float3 tangent;
	float3 bitangent;
	float3 normal;

	float3 omega_i;

	MaterialDiffuse material;
	float3          albedo;

    __device__ void init(int bounce, bool entering_material, int material_id, float2 tex_coord, const LOD & lod) {
        material = material_as_diffuse(material_id);
		albedo = sample_albedo(bounce, material.diffuse, material.texture_id, tex_coord, lod);
    }

	__device__ void attenuate(int bounce, int pixel_index, float3 & throughput, float distance) {
		if (bounce > 0) {
			throughput *= albedo;
		} else if (config.enable_albedo || config.enable_svgf) {
			frame_buffer_albedo[pixel_index] = make_float4(albedo);
		}
	}

	__device__ bool eval(const float3 & to_light, float cos_theta_o, float3 & bsdf, float & pdf) const {
		if (cos_theta_o <= 0.0f) return false;

		bsdf = make_float3(cos_theta_o * ONE_OVER_PI);
		pdf  = cos_theta_o * ONE_OVER_PI;

		return pdf_is_valid(pdf);
	}

	__device__ bool sample(float3 & throughput, float3 & direction_out, float & pdf) const {
		float2 rand_brdf = random<SampleDimension::BRDF>(pixel_index, bounce, sample_index);
		float3 omega_o = sample_cosine_weighted_direction(rand_brdf.x, rand_brdf.y);

		direction_out = local_to_world(omega_o, tangent, bitangent, normal);
		pdf = omega_o.z * ONE_OVER_PI;

		return pdf_is_valid(pdf);
	}

	__device__ bool is_mis_eligable() const {
		return true;
	}
};

struct BSDFPlastic {
	static constexpr bool HAS_ALBEDO = true;

	int pixel_index;
	int bounce;
	int sample_index;

	float3 tangent;
	float3 bitangent;
	float3 normal;

	float3 omega_i;

	MaterialPlastic material;
	float3          albedo;

	static constexpr float ETA = 1.0f / 1.5f;

	// The Total Internal Reflection compensation factor is calculated as
	// the hemispherical integral of fresnel * cos(theta)
	// This integral has a closed form and can be calculated as follows:
	//
	// float fresnel_first_moment(float ior) {
	//      double n  = ior;
	//      double n2 = n * n;
	//      double n4 = n2 * n2;
	//
	//      auto sq = [](auto x) { return x * x; };
	//      auto cb = [](auto x) { return x * x * x; };
	//
	//      auto a = (n - 1.0)*(3.0*n + 1.0) / (6.0 * sq(n + 1.0));
	//      auto b = (n2*sq(n2 - 1.0) / cb(n2 + 1.0)) * log((n - 1.0) / (n + 1.0));
	//      auto c = (2.0*cb(n)*(n2 + 2.0*n - 1.0)) / ((n2 + 1.0) * (n4 - 1.0));
	//      auto d = (8.0*n4*(n4 + 1.0) / ((n2 + 1.0) * sq(n4 - 1.0))) * log(n);
	//
	//      return float(0.5 + a + b - c + d);
	//	}
	//
	//	TIR_COMPENSATION = 1.0f - (1.0f - fresnel_first_moment(ior)) / (ior * ior);
	//
	// TIR_COMPENSATION has been precalculated for ior = 1.5f. Unfortunately constexpr
	// does not work with math functions, so the value has to be hardcoded.
	static constexpr float TIR_COMPENSATION = 0.596345782f;

	__device__ void init(int bounce, bool entering_material, int material_id, float2 tex_coord, const LOD & lod) {
		material = material_as_plastic(material_id);
		albedo = sample_albedo(bounce, material.diffuse, material.texture_id, tex_coord, lod);
	}

	__device__ void attenuate(int bounce, int pixel_index, float3 & ray_throughput, float distance) {
		if (bounce == 0 && (config.enable_albedo || config.enable_svgf)) {
			frame_buffer_albedo[pixel_index] = make_float4(1.0f);
		}
	}

	__device__ bool eval(const float3 & to_light, float cos_theta_o, float3 & bsdf, float & pdf) const {
		if (cos_theta_o <= 0.0f) return false;

		float3 omega_o = world_to_local(to_light, tangent, bitangent, normal);
		float3 omega_m = normalize(omega_i + omega_o);

		// Specular component
		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		float F  = fresnel_dielectric(dot(omega_i, omega_m), ETA);
		float D  = ggx_D (omega_m, alpha_x, alpha_y);
		float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		float3 brdf_specular = make_float3(F * G2 * D / (4.0f * omega_i.z));

		// Diffuse component
		float F_i = fresnel_dielectric(omega_i.z, ETA);
		float F_o = fresnel_dielectric(omega_o.z, ETA);

		float3 brdf_diffuse = ETA*ETA * (1.0f - F_i) * (1.0f - F_o) * albedo * ONE_OVER_PI / (1.0f - albedo * TIR_COMPENSATION) * omega_o.z;

		float pdf_specular = G1 * D / (4.0f * omega_i.z);
		float pdf_diffuse  = omega_o.z * ONE_OVER_PI;

		pdf  = lerp(pdf_diffuse, pdf_specular, F_i);
		bsdf = brdf_specular + brdf_diffuse; // BRDF * cos(theta_o)

		return pdf_is_valid(pdf);
	}

	__device__ bool sample(float3 & throughput, float3 & direction_out, float & pdf) const {
		float  rand_fresnel = random<SampleDimension::RUSSIAN_ROULETTE>(pixel_index, bounce, sample_index).y;
		float2 rand_brdf    = random<SampleDimension::BRDF>            (pixel_index, bounce, sample_index);

		float F_i = fresnel_dielectric(omega_i.z, ETA);

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		float3 omega_m;
		float3 omega_o;
		if (rand_fresnel < F_i) {
			// Sample specular component
			omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_brdf.x, rand_brdf.y);
			omega_o = reflect(-omega_i, omega_m);
		} else {
			// Sample diffuse component
			omega_o = sample_cosine_weighted_direction(rand_brdf.x, rand_brdf.y);
			omega_m = normalize(omega_i + omega_o);
		}

		if (omega_m.z < 0.0f) return false; // Wrong hemisphere

		// Specular component
		float F  = fresnel_dielectric(dot(omega_i, omega_m), ETA);
		float D  = ggx_D (omega_m, alpha_x, alpha_y);
		float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		float3 brdf_specular = make_float3(F * G2 * D / (4.0f * omega_i.z));

		// Diffuse component
		float F_o = fresnel_dielectric(omega_o.z, ETA);

		float3 brdf_diffuse = ETA*ETA * (1.0f - F_i) * (1.0f - F_o) * albedo * ONE_OVER_PI / (1.0f - albedo * TIR_COMPENSATION) * omega_o.z;

		float pdf_specular = G1 * D / (4.0f * omega_i.z);
		float pdf_diffuse  = omega_o.z * ONE_OVER_PI;
		pdf = lerp(pdf_diffuse, pdf_specular, F_i);

		throughput *= (brdf_specular + brdf_diffuse) / pdf; // BRDF * cos(theta) / pdf

		direction_out = local_to_world(omega_o, tangent, bitangent, normal);

		return pdf_is_valid(pdf);
	}

	__device__ bool is_mis_eligable() const {
		return true;
	}
};

struct BSDFDielectric {
	static constexpr bool HAS_ALBEDO = false;

	int pixel_index;
	int bounce;
	int sample_index;

	float3 tangent;
	float3 bitangent;
	float3 normal;

	float3 omega_i;

	MaterialDielectric material;

	float eta;

	__device__ void init(int bounce, bool entering_material, int material_id, float2 tex_coord, const LOD & lod) {
		material = material_as_dielectric(material_id);

		eta = entering_material ? 1.0f / material.index_of_refraction : material.index_of_refraction;
	}

	__device__ void attenuate(int bounce, int pixel_index, float3 & throughput, float distance) {
		frame_buffer_albedo[pixel_index] = make_float4(1.0f);

		// Lambert-Beer Law
		// NOTE: does not take into account e.g. nested dielectrics or diffuse inside dielectric!
		throughput.x *= expf(material.negative_absorption.x * distance);
		throughput.y *= expf(material.negative_absorption.y * distance);
		throughput.z *= expf(material.negative_absorption.z * distance);
	}

	__device__ bool eval(const float3 & to_light, float cos_theta_o, float3 & bsdf, float & pdf) const {
		float3 omega_o = world_to_local(to_light, tangent, bitangent, normal);

		bool reflected = omega_o.z >= 0.0f; // Same sign means reflection, alternate signs means transmission

		float3 omega_m;
		if (reflected) {
			omega_m = normalize(omega_i + omega_o);
		} else {
			omega_m = normalize(eta * omega_i + omega_o);
		}
		omega_m *= sign(omega_m.z);

		float i_dot_m = abs_dot(omega_i, omega_m);
		float o_dot_m = abs_dot(omega_o, omega_m);

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		float F  = fresnel_dielectric(i_dot_m, eta);
		float D  = ggx_D (omega_m, alpha_x, alpha_y);
		float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		if (reflected) {
			pdf = F * G1 * D / (4.0f * omega_i.z);

			bsdf = make_float3(F * G2 * D / (4.0f * omega_i.z)); // BRDF times cos(theta_o)
		} else {
			if (F >= 0.999f) return false; // TIR, no transmission possible

			pdf = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));

			bsdf = eta * eta * make_float3((1.0f - F) * G2 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m))); // BRDF times cos(theta_o)
		}

		return pdf_is_valid(pdf);
	}

	__device__ bool sample(float3 & throughput, float3 & direction_out, float & pdf) const {
		float  rand_fresnel = random<SampleDimension::RUSSIAN_ROULETTE>(pixel_index, bounce, sample_index).y;
		float2 rand_brdf    = random<SampleDimension::BRDF>            (pixel_index, bounce, sample_index);

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		float3 omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_brdf.x, rand_brdf.y);

		float F = fresnel_dielectric(abs_dot(omega_i, omega_m), eta);
		bool reflected = rand_fresnel < F;

		float3 omega_o;
		if (reflected) {
			omega_o = 2.0f * dot(omega_i, omega_m) * omega_m - omega_i;
		} else {
			float k = 1.0f - eta*eta * (1.0f - square(dot(omega_i, omega_m)));
			omega_o = (eta * abs_dot(omega_i, omega_m) - sqrtf(k)) * omega_m - eta * omega_i;
		}

		if (reflected ^ (omega_o.z >= 0.0f)) return false; // Hemisphere check: reflection should have positive z, transmission negative z

		float D  = ggx_D (omega_m, alpha_x, alpha_y);
		float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		float i_dot_m = abs_dot(omega_i, omega_m);
		float o_dot_m = abs_dot(omega_o, omega_m);

		if (reflected) {
			pdf = F * G1 * D / (4.0f * omega_i.z);
		} else {
			pdf = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));

			throughput *= eta*eta; // Account for solid angle compression
		}

		throughput *= G2 / G1; // BRDF * cos(theta_o) / pdf (same for reflection and transmission)

		direction_out = local_to_world(omega_o, tangent, bitangent, normal);

		return pdf_is_valid(pdf);
	}

	__device__ bool is_mis_eligable() const {
		return material.roughness >= ROUGHNESS_CUTOFF;
	}
};

struct BSDFConductor {
	static constexpr bool HAS_ALBEDO = false;

	int pixel_index;
	int bounce;
	int sample_index;

	float3 tangent;
	float3 bitangent;
	float3 normal;

	float3 omega_i;

	MaterialConductor material;

	__device__ void init(int bounce, bool entering_material, int material_id, float2 tex_coord, const LOD & lod) {
		material = material_as_conductor(material_id);
	}

	__device__ void attenuate(int bounce, int pixel_index, float3 & throughput, float distance) {
		if (bounce == 0 && (config.enable_albedo || config.enable_svgf)) {
			frame_buffer_albedo[pixel_index] = make_float4(1.0f);
		}
	}

	__device__ bool eval(const float3 & to_light, float cos_theta_o, float3 & bsdf, float & pdf) const {
		if (cos_theta_o <= 0.0f) return false;

		float3 omega_o = world_to_local(to_light, tangent, bitangent, normal);
		float3 omega_m = normalize(omega_o + omega_i);

		float o_dot_m = dot(omega_o, omega_m);
		if (o_dot_m <= 0.0f) return false;

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		float3 F  = fresnel_conductor(o_dot_m, material.eta, material.k);
		float  D  = ggx_D (omega_m, alpha_x, alpha_y);
		float  G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float  G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		pdf  =     G1 * D / (4.0f * omega_i.z);
		bsdf = F * G2 * D / (4.0f * omega_i.z); // BRDF * cos(theta_o)

		return pdf_is_valid(pdf);
	}

	__device__ bool sample(float3 & throughput, float3 & direction_out, float & pdf) const {
		float2 rand_brdf = random<SampleDimension::BRDF>(pixel_index, bounce, sample_index);

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		float3 omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_brdf.x, rand_brdf.y);
		float3 omega_o = reflect(-omega_i, omega_m);

		float o_dot_m = dot(omega_o, omega_m);
		if (o_dot_m <= 0.0f) return false;

		float3 F  = fresnel_conductor(o_dot_m, material.eta, material.k);
		float  D  = ggx_D (omega_m, alpha_x, alpha_y);
		float  G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float  G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		pdf = G1 * D / (4.0f * omega_i.z);

		throughput *= F * G2 / G1; // BRDF * cos(theta_o) / pdf

		direction_out = local_to_world(omega_o, tangent, bitangent, normal);

		return pdf_is_valid(pdf);
	}

	__device__ bool is_mis_eligable() const {
		return material.roughness >= ROUGHNESS_CUTOFF;
	}
};
