#pragma once
#include "KullaConty.h"
#include "Sampling.h"
#include "Material.h"
#include "RayCone.h"
#include "AOV.h"

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

    __device__ void init(int bounce, bool entering_material, int material_id) {
        material = material_as_diffuse(material_id);
    }

	__device__ void calc_albedo(int bounce, int pixel_index, float3 & throughput, float2 tex_coord, const TextureLOD & lod) {
		albedo = sample_albedo(bounce, material.diffuse, material.texture_id, tex_coord, lod);

		if (bounce == 0) {
			aov_framebuffer_set(AOVType::ALBEDO, pixel_index, make_float4(albedo));
		}
		if (!(config.enable_svgf && bounce == 0)) {
			throughput *= albedo;
		}
	}

	__device__ bool eval(const float3 & to_light, float cos_theta_o, float3 & bsdf, float & pdf) const {
		if (cos_theta_o <= 0.0f) return false;

		bsdf = make_float3(cos_theta_o * ONE_OVER_PI);
		pdf  = cos_theta_o * ONE_OVER_PI;

		return pdf_is_valid(pdf);
	}

	__device__ bool sample(float3 & throughput, int & medium_id, float3 & direction_out, float & pdf) const {
		float2 rand_brdf = random<SampleDimension::BSDF_0>(pixel_index, bounce, sample_index);
		float3 omega_o = sample_cosine_weighted_direction(rand_brdf.x, rand_brdf.y);

		direction_out = local_to_world(omega_o, tangent, bitangent, normal);
		pdf = omega_o.z * ONE_OVER_PI;

		return pdf_is_valid(pdf);
	}

	__device__ bool has_texture() const {
		return material.texture_id != INVALID;
	}

	__device__ bool allow_nee() const {
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
	//  }
	//
	//	TIR_COMPENSATION = 1.0f - (1.0f - fresnel_first_moment(ior)) / (ior * ior);
	//
	// TIR_COMPENSATION has been precalculated for ior = 1.5f. Unfortunately constexpr
	// does not work with math functions, so the value has to be hardcoded.
	static constexpr float TIR_COMPENSATION = 0.596345782f;

	__device__ void init(int bounce, bool entering_material, int material_id) {
		material = material_as_plastic(material_id);
	}

	__device__ void calc_albedo(int bounce, int pixel_index, float3 & ray_throughput, float2 tex_coord, const TextureLOD & lod) {
		albedo = sample_albedo(bounce, material.diffuse, material.texture_id, tex_coord, lod);

		if (bounce == 0) {
			aov_framebuffer_set(AOVType::ALBEDO, pixel_index, make_float4(albedo));
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

	__device__ bool sample(float3 & throughput, int & medium_id, float3 & direction_out, float & pdf) const {
		float  rand_fresnel = random<SampleDimension::BSDF_0>(pixel_index, bounce, sample_index).x;
		float2 rand_brdf    = random<SampleDimension::BSDF_1>(pixel_index, bounce, sample_index);

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

	__device__ bool has_texture() const {
		return material.texture_id != INVALID;
	}

	__device__ bool allow_nee() const {
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

	__device__ void init(int bounce, bool entering_material, int material_id) {
		material = material_as_dielectric(material_id);

		eta = entering_material ? 1.0f / material.ior : material.ior;
	}

	__device__ void calc_albedo(int bounce, int pixel_index, float3 & throughput, float2 tex_coord, const TextureLOD & lod) {
		// NO-OP
	}

	__device__ bool eval(const float3 & to_light, float cos_theta_o, float3 & bsdf, float & pdf) const {
		float3 omega_o = world_to_local(to_light, tangent, bitangent, normal);

		bool reflected = omega_o.z >= 0.0f; // Positive sign means reflection, negative sign means transmission

		float3 omega_m;
		if (reflected) {
			omega_m = normalize(omega_i + omega_o);
		} else {
			omega_m = normalize(eta * omega_i + omega_o);
		}
		if (omega_m.z <= 0.0f) return false;
		
		float i_dot_m = abs_dot(omega_i, omega_m);
		float o_dot_m = abs_dot(omega_o, omega_m);

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		float F  = fresnel_dielectric(i_dot_m, eta);
		float D  = ggx_D (omega_m, alpha_x, alpha_y);
		float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		bool entering_material = eta < 1.0f;

		float F_avg = average_fresnel(material.ior);
		if (!entering_material) {
			F_avg = 1.0f - (1.0f - F_avg) / square(material.ior);
		}
		
		float x = kulla_conty_x(material.ior, material.roughness);
		float ratio = (entering_material ? x : (1.0f - x)) * (1.0f - F_avg);

		float E_i = dielectric_directional_albedo(material.ior, material.roughness, omega_i.z, entering_material);

		float bsdf_single;
		float bsdf_multi;

		float pdf_single;
		float pdf_multi;

		if (reflected) {
			bsdf_single = F * G2 * D / (4.0f * omega_i.z); // BRDF times cos(theta_o)
			pdf_single  = F * G1 * D / (4.0f * omega_i.z);

			float E_o   = dielectric_directional_albedo(material.ior, material.roughness, omega_o.z, entering_material);
			float E_avg = dielectric_albedo(material.ior, material.roughness, entering_material);

			bsdf_multi = (1.0f - ratio) * fabsf(omega_o.z) * kulla_conty_multiscatter(E_i, E_o, E_avg);
			pdf_multi  = (1.0f - ratio) * fabsf(omega_o.z) * ONE_OVER_PI;
		} else {
			bsdf_single = (1.0f - F) * G2 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m) * square(eta)); // BRDF times cos(theta_o)
			pdf_single  = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));

			float E_o   = dielectric_directional_albedo(material.ior, material.roughness, omega_o.z, !entering_material);
			float E_avg = dielectric_albedo(material.ior, material.roughness, !entering_material);

			bsdf_multi = ratio * fabsf(omega_o.z) * kulla_conty_multiscatter(E_i, E_o, E_avg);
			pdf_multi  = ratio * fabsf(omega_o.z) * ONE_OVER_PI;
		}

		bsdf = make_float3(bsdf_single + bsdf_multi);

		pdf = lerp(pdf_multi, pdf_single, E_i);
		return pdf_is_valid(pdf);
	}

	__device__ bool sample(float3 & throughput, int & medium_id, float3 & direction_out, float & pdf) const {
		float2 rand_bsdf_0 = random<SampleDimension::BSDF_0>(pixel_index, bounce, sample_index);
		float2 rand_bsdf_1 = random<SampleDimension::BSDF_1>(pixel_index, bounce, sample_index);

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		bool entering_material = eta < 1.0f;

		float E_i = dielectric_directional_albedo(material.ior, material.roughness, omega_i.z, entering_material);
		
		float F_avg = average_fresnel(material.ior);
		if (!entering_material) {
			F_avg = 1.0f - (1.0f - F_avg) / square(material.ior);
		}
		
		float x = kulla_conty_x(material.ior, material.roughness);
		float ratio = (entering_material ? x : (1.0f - x)) * (1.0f - F_avg);

		float F;
		bool reflected;

		float3 omega_m;
		float3 omega_o;
		if (rand_bsdf_0.x < E_i) {
			// Sample single scatter component
			omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_bsdf_1.x, rand_bsdf_1.y);

			F = fresnel_dielectric(abs_dot(omega_i, omega_m), eta);
			reflected = rand_bsdf_0.y < F;

			if (reflected) {
				omega_o = 2.0f * dot(omega_i, omega_m) * omega_m - omega_i;
			} else {
				float k = 1.0f - eta*eta * (1.0f - square(dot(omega_i, omega_m)));
				omega_o = (eta * dot(omega_i, omega_m) - safe_sqrt(k)) * omega_m - eta * omega_i;
			}
		} else {
			// Sample multiple scatter component
			omega_o = sample_cosine_weighted_direction(rand_bsdf_1.x, rand_bsdf_1.y);

			reflected = rand_bsdf_0.y > ratio;

			if (reflected) {
				omega_m = normalize(omega_i + omega_o);
			} else {
				omega_o = -omega_o;
				omega_m = normalize(eta * omega_i + omega_o);
			}
			omega_m *= sign(omega_m.z);

			F = fresnel_dielectric(abs_dot(omega_i, omega_m), eta);
		}

		if (reflected ^ (omega_o.z >= 0.0f)) return false; // Hemisphere check: reflection should have positive z, transmission negative z

		float D  = ggx_D (omega_m, alpha_x, alpha_y);
		float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		float i_dot_m = abs_dot(omega_i, omega_m);
		float o_dot_m = abs_dot(omega_o, omega_m);

		float bsdf_single;
		float bsdf_multi;

		float pdf_single;
		float pdf_multi;

		if (reflected) {
			bsdf_single = F * G2 * D / (4.0f * omega_i.z); // BRDF times cos(theta_o)
			pdf_single  = F * G1 * D / (4.0f * omega_i.z);

			float E_o   = dielectric_directional_albedo(material.ior, material.roughness, omega_o.z, entering_material);
			float E_avg = dielectric_albedo(material.ior, material.roughness, entering_material);

			bsdf_multi = (1.0f - ratio) * fabsf(omega_o.z) * kulla_conty_multiscatter(E_i, E_o, E_avg);
			pdf_multi  = (1.0f - ratio) * fabsf(omega_o.z) * ONE_OVER_PI;
		} else {
			bsdf_single = (1.0f - F) * G2 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m) * square(eta)); // BRDF times cos(theta_o)
			pdf_single  = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));

			float E_o   = dielectric_directional_albedo(material.ior, material.roughness, omega_o.z, !entering_material);
			float E_avg = dielectric_albedo(material.ior, material.roughness, !entering_material);

			bsdf_multi = ratio * fabsf(omega_o.z) * kulla_conty_multiscatter(E_i, E_o, E_avg);
			pdf_multi  = ratio * fabsf(omega_o.z) * ONE_OVER_PI;

			// Update the Medium based on whether we are transmitting into or out of the Material
			if (entering_material) {
				medium_id = material.medium_id;
			} else {
				medium_id = INVALID;
			}
		}

		pdf = lerp(pdf_multi, pdf_single, E_i);
		throughput *= (bsdf_single + bsdf_multi) / pdf;

		direction_out = local_to_world(omega_o, tangent, bitangent, normal);

		return pdf_is_valid(pdf);
	}

	__device__ bool has_texture() const {
		return false;
	}

	__device__ bool allow_nee() const {
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

	__device__ void init(int bounce, bool entering_material, int material_id) {
		material = material_as_conductor(material_id);
	}

	__device__ void calc_albedo(int bounce, int pixel_index, float3 & throughput, float2 tex_coord, const TextureLOD & lod) {
		// NO-OP
	}

	__device__ bool eval(const float3 & to_light, float cos_theta_o, float3 & bsdf, float & pdf) const {
		if (cos_theta_o <= 0.0f) return false;

		float3 omega_o = world_to_local(to_light, tangent, bitangent, normal);
		float3 omega_m = normalize(omega_o + omega_i);

		float o_dot_m = dot(omega_o, omega_m);
		if (o_dot_m <= 0.0f) return false;

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		// Single scatter lobe
		float3 F  = fresnel_conductor(o_dot_m, material.eta, material.k);
		float  D  = ggx_D (omega_m, alpha_x, alpha_y);
		float  G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float  G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		float3 brdf_single = F * G2 * D / (4.0f * omega_i.z); // BRDF * cos(theta_o)
		float  pdf_single  =     G1 * D / (4.0f * omega_i.z);

		// Multi scatter lobe
		float E_i   = conductor_directional_albedo(material.roughness, omega_i.z);
		float E_o   = conductor_directional_albedo(material.roughness, omega_o.z);
		float E_avg = conductor_albedo(material.roughness);

		float3 F_avg = average_fresnel(material.eta, material.k);
		float3 F_ms  = fresnel_multiscatter(F_avg, E_avg);

		float3 brdf_multi = F_ms * kulla_conty_multiscatter(E_i, E_o, E_avg) * omega_o.z;
		float  pdf_multi  = omega_o.z * ONE_OVER_PI;

		bsdf = brdf_single + brdf_multi;

		pdf = lerp(pdf_multi, pdf_single, E_i);
		return pdf_is_valid(pdf);
	}

	__device__ bool sample(float3 & throughput, int & medium_id, float3 & direction_out, float & pdf) const {
		float2 rand_brdf_0 = random<SampleDimension::BSDF_0>(pixel_index, bounce, sample_index);
		float2 rand_brdf_1 = random<SampleDimension::BSDF_1>(pixel_index, bounce, sample_index);

		float alpha_x = material.roughness;
		float alpha_y = material.roughness;

		float E_i = conductor_directional_albedo(material.roughness, omega_i.z);

		float3 omega_m;
		float3 omega_o;
		if (rand_brdf_0.x < E_i) {
			// Sample single scatter component
			omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_brdf_1.x, rand_brdf_1.y);
			omega_o = reflect(-omega_i, omega_m);
		} else {
			// Sample multiple scatter component
			omega_o = sample_cosine_weighted_direction(rand_brdf_1.x, rand_brdf_1.y);
			omega_m = normalize(omega_i + omega_o);
		}

		float o_dot_m = dot(omega_o, omega_m);
		if (o_dot_m <= 0.0f || omega_o.z < 0.0f) return false;

		float3 F  = fresnel_conductor(o_dot_m, material.eta, material.k);
		float  D  = ggx_D (omega_m, alpha_x, alpha_y);
		float  G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float  G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		float E_o   = conductor_directional_albedo(material.roughness, omega_o.z);
		float E_avg = conductor_albedo(material.roughness);

		float3 F_avg = average_fresnel(material.eta, material.k);
		float3 F_ms  = fresnel_multiscatter(F_avg, E_avg);

		float3 brdf_single = F * G2 * D / (4.0f * omega_i.z);
		float3 brdf_multi  = F_ms * kulla_conty_multiscatter(E_i, E_o, E_avg) * omega_o.z;

		float pdf_single = G1 * D / (4.0f * omega_i.z);
		float pdf_multi  = omega_o.z * ONE_OVER_PI;
		pdf = lerp(pdf_multi, pdf_single, E_i);

		throughput *= (brdf_single + brdf_multi) / pdf;

		direction_out = local_to_world(omega_o, tangent, bitangent, normal);

		return pdf_is_valid(pdf);
	}

	__device__ bool has_texture() const {
		return false;
	}

	__device__ bool allow_nee() const {
		return material.roughness >= ROUGHNESS_CUTOFF;
	}
};
