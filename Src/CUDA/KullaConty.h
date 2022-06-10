#pragma once
#include "Material.h"

__device__ Texture<float> lut_dielectric_directional_albedo_enter;
__device__ Texture<float> lut_dielectric_directional_albedo_leave;
__device__ Texture<float> lut_dielectric_albedo_enter;
__device__ Texture<float> lut_dielectric_albedo_leave;

__device__ Texture<float> lut_conductor_directional_albedo;
__device__ Texture<float> lut_conductor_albedo;

__device__ inline float3 fresnel_multiscatter(const float3 & F_avg, float E_avg) {
	return F_avg*F_avg * E_avg / (make_float3(1.0f) - F_avg * (1.0f - E_avg));
}

__device__ inline float dielectric_directional_albedo(float ior, float linear_roughness, float cos_theta, bool entering_material) {
	ior = remap(ior, LUT_MIN_IOR, LUT_MAX_IOR, 0.0f, 1.0f);
	cos_theta = fabsf(cos_theta);

	return (entering_material ? lut_dielectric_directional_albedo_enter : lut_dielectric_directional_albedo_leave).get(ior, linear_roughness, cos_theta);
}

__device__ inline float dielectric_albedo(float ior, float linear_roughness, bool entering_material) {
	ior = remap(ior, LUT_MIN_IOR, LUT_MAX_IOR, 0.0f, 1.0f);

	return (entering_material ? lut_dielectric_albedo_enter : lut_dielectric_albedo_leave).get(ior, linear_roughness);
}

__device__ inline float conductor_directional_albedo(float linear_roughness, float cos_theta) {
	cos_theta = fabsf(cos_theta);

	return lut_conductor_directional_albedo.get(linear_roughness, cos_theta);
}

__device__ inline float conductor_albedo(float linear_roughness) {
	return lut_conductor_albedo.get(linear_roughness);
}

__device__ inline float kulla_conty_multiscatter(float E_i, float E_o, float E_avg) {
	assert(0.0f <= E_i   && E_i   <= 1.0f);
	assert(0.0f <= E_o   && E_o   <= 1.0f);
	assert(0.0f <= E_avg && E_avg <= 1.0f);

	return (1.0f - E_i) * (1.0f - E_o) / fmaxf(0.0001f, PI * (1.0f - E_avg));
}

__device__ inline float kulla_conty_x(float ior, float linear_roughness) { // TOOD: OPTIMIZE
	float E_avg_eta     = dielectric_albedo(ior, linear_roughness, true);
	float E_avg_eta_inv = dielectric_albedo(ior, linear_roughness, false);

	return (1.0f - E_avg_eta_inv) / fmaxf(0.0001f, 2.0f - E_avg_eta - E_avg_eta_inv);
}

__device__ inline float sample_dielectric(int thread_index, int sample_index, float linear_roughness, float eta, float3 omega_i) {
	float  rand_fresnel = random<SampleDimension::RUSSIAN_ROULETTE>(thread_index, 0, sample_index).y;
	float2 rand_brdf    = random<SampleDimension::BSDF_0>          (thread_index, 0, sample_index);

	float alpha_x = roughness_to_alpha(linear_roughness);
	float alpha_y = roughness_to_alpha(linear_roughness);

	float3 omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_brdf.x, rand_brdf.y);

	float F = fresnel_dielectric(abs_dot(omega_i, omega_m), eta);
	bool reflected = rand_fresnel < F;

	float3 omega_o;
	if (reflected) {
		omega_o = 2.0f * dot(omega_i, omega_m) * omega_m - omega_i;
	} else {
		float k = 1.0f - eta*eta * (1.0f - square(dot(omega_i, omega_m)));
		omega_o = (eta * abs_dot(omega_i, omega_m) - safe_sqrt(k)) * omega_m - eta * omega_i;
	}

	if (reflected ^ (omega_o.z >= 0.0f)) return 0.0f; // Hemisphere check: reflection should have positive z, transmission negative z

	float D  = ggx_D (omega_m, alpha_x, alpha_y);
	float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
	float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

	float i_dot_m = abs_dot(omega_i, omega_m);
	float o_dot_m = abs_dot(omega_o, omega_m);

	float weight = G2 / G1; // BRDF * cos(theta_o) / pdf (same for reflection and transmission)

	float pdf;
	if (reflected) {
		pdf = F * G1 * D / (4.0f * omega_i.z);
	} else {
		pdf = (1.0f - F) * G1 * D * i_dot_m * o_dot_m / (omega_i.z * square(eta * i_dot_m + o_dot_m));
	}
	return pdf_is_valid(pdf) ? weight : 0.0f;
}

__device__ inline int lut_dielectric_index(int i, int r, int c) {
	return i + r * LUT_DIELECTRIC_DIM_IOR + c * LUT_DIELECTRIC_DIM_IOR * LUT_DIELECTRIC_DIM_ROUGHNESS;
}

__device__ inline float lut_dielectric_map_ior(int index_ior) {
#if 1
	return remap((float(index_ior) + 0.5f) / float(LUT_DIELECTRIC_DIM_IOR), 0.0f, 1.0f, LUT_MIN_IOR, LUT_MAX_IOR);
#else
	return 0.0001f + remap((float(index_ior)) / float(LUT_DIELECTRIC_DIM_IOR-1), 0.0f, 1.0f, LUT_MIN_IOR, LUT_MAX_IOR);
#endif
}

__device__ inline float lut_dielectric_map_roughness(int index_roughness) {
#if 1
	return (float(index_roughness) + 0.5f) / float(LUT_DIELECTRIC_DIM_ROUGHNESS);
#else
	return fmaxf(1e-6f, square((float(index_roughness)) / float(LUT_DIELECTRIC_DIM_ROUGHNESS-1)));
#endif
}

__device__ inline float lut_dielectric_map_cos_theta(int index_cos_theta) {
#if 1
	return (float(index_cos_theta) + 0.5f) / float(LUT_DIELECTRIC_DIM_COS_THETA);
#else
	return clamp((float(index_cos_theta)) / float(LUT_DIELECTRIC_DIM_COS_THETA-1), 0.001f, 0.999f);
#endif
}

extern "C" __global__ void kernel_integrate_dielectric(bool entering_material, float * lut_directional_albedo) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= LUT_DIELECTRIC_DIM_IOR * LUT_DIELECTRIC_DIM_ROUGHNESS * LUT_DIELECTRIC_DIM_COS_THETA) return;

	int i = (thread_index)                                                           % LUT_DIELECTRIC_DIM_IOR;
	int r = (thread_index / (LUT_DIELECTRIC_DIM_IOR))                                % LUT_DIELECTRIC_DIM_ROUGHNESS;
	int c = (thread_index / (LUT_DIELECTRIC_DIM_IOR * LUT_DIELECTRIC_DIM_ROUGHNESS)) % LUT_DIELECTRIC_DIM_COS_THETA;

	float ior = lut_dielectric_map_ior(i);
	float eta = entering_material ? 1.0f / ior : ior;

	float linear_roughness = lut_dielectric_map_roughness(r);

	float cos_theta = lut_dielectric_map_cos_theta(c);
	float sin_theta = safe_sqrt(1.0f - square(cos_theta));
	float3 omega_i = make_float3(sin_theta, 0.0f, cos_theta);

	float avg = 0.0f;
	constexpr int NUM_SAMPLES = 100000;

	for (int s = 0; s < NUM_SAMPLES; s++) {
		float weight = sample_dielectric(thread_index, s, linear_roughness, eta, omega_i);
		avg = online_average(avg, weight, s + 1);
	}

	lut_directional_albedo[thread_index] = avg;
}

extern "C" __global__ void kernel_average_dielectric(const float * lut_directional_albedo, float * lut_albedo) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= LUT_DIELECTRIC_DIM_IOR * LUT_DIELECTRIC_DIM_ROUGHNESS) return;

	int i = (thread_index)               % LUT_DIELECTRIC_DIM_IOR;
	int r = (thread_index / LUT_DIELECTRIC_DIM_IOR) % LUT_DIELECTRIC_DIM_ROUGHNESS;

	float avg = 0.0f;

	for (int c = 0; c < LUT_DIELECTRIC_DIM_COS_THETA; c++) {
		float cos_theta = lut_dielectric_map_cos_theta(c);

		int index = lut_dielectric_index(i, r, c);
		avg = online_average(avg, lut_directional_albedo[index] * cos_theta, c + 1);
	}

	lut_albedo[thread_index] = 2.0f * avg;
}

__device__ inline float sample_conductor(int thread_index, int sample_index, float linear_roughness, float3 omega_i) {
		float2 rand_brdf = random<SampleDimension::BSDF_0>(thread_index, 0, sample_index);

		float alpha_x = roughness_to_alpha(linear_roughness);
		float alpha_y = roughness_to_alpha(linear_roughness);

		float3 omega_m = sample_visible_normals_ggx(omega_i, alpha_x, alpha_y, rand_brdf.x, rand_brdf.y);
		float3 omega_o = reflect(-omega_i, omega_m);

		if (dot(omega_o, omega_m) <= 0.0f || omega_o.z <= 0.0f) return 0.0f;

		float D  = ggx_D (omega_m, alpha_x, alpha_y);
		float G1 = ggx_G1(omega_i, alpha_x, alpha_y);
		float G2 = ggx_G2(omega_o, omega_i, omega_m, alpha_x, alpha_y);

		float weight = G2 / G1; // BRDF * cos(theta_o) / pdf (NOTE: Fresnel factor not included!)

		float pdf = G1 * D / (4.0f * omega_i.z);
		return pdf_is_valid(pdf) ? weight : 0.0f;
	}

__device__ inline int lut_conductor_index(int r, int c) {
	return r + c * LUT_CONDUCTOR_DIM_ROUGHNESS;
}

__device__ inline float lut_conductor_map_roughness(int index_roughness) {
#if 1
	return (float(index_roughness) + 0.5f) / float(LUT_CONDUCTOR_DIM_ROUGHNESS);
#else
	return fmaxf(1e-6f, square((float(index_roughness)) / float(LUT_CONDUCTOR_DIM_ROUGHNESS-1)));
#endif
}

__device__ inline float lut_conductor_map_cos_theta(int index_cos_theta) {
#if 1
	return (float(index_cos_theta) + 0.5f) / float(LUT_CONDUCTOR_DIM_COS_THETA);
#else
	return clamp((float(index_cos_theta)) / float(LUT_CONDUCTOR_DIM_COS_THETA-1), 0.001f, 0.999f);
#endif
}

extern "C" __global__ void kernel_integrate_conductor(float * lut_directional_albedo) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= LUT_CONDUCTOR_DIM_ROUGHNESS * LUT_CONDUCTOR_DIM_COS_THETA) return;

	int r = (thread_index)                                 % LUT_CONDUCTOR_DIM_ROUGHNESS;
	int c = (thread_index / (LUT_CONDUCTOR_DIM_ROUGHNESS)) % LUT_CONDUCTOR_DIM_COS_THETA;

	float linear_roughness = lut_conductor_map_roughness(r);

	float cos_theta = lut_conductor_map_cos_theta(c);
	float sin_theta = safe_sqrt(1.0f - square(cos_theta));
	float3 omega_i = make_float3(sin_theta, 0.0f, cos_theta);

	float avg = 0.0f;
	constexpr int NUM_SAMPLES = 100000;

	for (int s = 0; s < NUM_SAMPLES; s++) {
		float weight = sample_conductor(thread_index, s, linear_roughness, omega_i);
		avg = online_average(avg, weight, s + 1);
	}

	lut_directional_albedo[thread_index] = avg;
}

extern "C" __global__ void kernel_average_conductor(const float * lut_directional_albedo, float * lut_albedo) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= LUT_CONDUCTOR_DIM_ROUGHNESS) return;

	int r = (thread_index) % LUT_CONDUCTOR_DIM_ROUGHNESS;

	float avg = 0.0f;

	for (int c = 0; c < LUT_CONDUCTOR_DIM_COS_THETA; c++) {
		float cos_theta = lut_conductor_map_cos_theta(c);

		int index = lut_conductor_index(r, c);
		avg = online_average(avg, lut_directional_albedo[index] * cos_theta, c + 1);
	}

	lut_albedo[thread_index] = 2.0f * avg;
}
