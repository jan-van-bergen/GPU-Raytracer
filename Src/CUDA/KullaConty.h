#pragma once
#include "Material.h"

__device__ Texture<float> lut_directional_albedo_enter;
__device__ Texture<float> lut_directional_albedo_leave;
__device__ Texture<float> lut_albedo_enter;
__device__ Texture<float> lut_albedo_leave;

/*
__device__ inline float fresnel_multiscatter(float F_avg, float E_avg) {
	return square(F_avg) * E_avg / (1.0f - F_avg * (1.0f - E_avg));
}
*/

__device__ inline float ggx_directional_albedo(float ior, float cos_theta, float roughness, bool entering_material) {
	ior = remap(ior, LUT_MIN_IOR, LUT_MAX_IOR, 0.0f, 1.0f);
	cos_theta = fabsf(cos_theta);
	roughness = sqrtf(roughness);

	return (entering_material ? lut_directional_albedo_enter : lut_directional_albedo_leave).get(ior, roughness, cos_theta);
}

__device__ inline float ggx_albedo(float ior, float roughness, bool entering_material) {
	ior = remap(ior, LUT_MIN_IOR, LUT_MAX_IOR, 0.0f, 1.0f);
	roughness = sqrtf(roughness);

	return (entering_material ? lut_albedo_enter : lut_albedo_leave).get(ior, roughness);
}

__device__ inline float kulla_conty_multiscatter(float E_i, float E_o, float E_avg) {
	assert(0.0f <= E_i   && E_i   <= 1.0f);
	assert(0.0f <= E_o   && E_o   <= 1.0f);
	assert(0.0f <= E_avg && E_avg <= 1.0f);

	return (1.0f - E_i) * (1.0f - E_o) / fmaxf(0.0001f, PI * (1.0f - E_avg));
}

__device__ inline float kulla_conty_x(float ior, float roughness) { // TOOD: OPTIMIZE
	float E_avg_eta     = ggx_albedo(ior, roughness, true);
	float E_avg_eta_inv = ggx_albedo(ior, roughness, false);

	return (1.0f - E_avg_eta_inv) / fmaxf(0.0001f, 2.0f - E_avg_eta - E_avg_eta_inv);
}

__device__ inline float sample_dielectric(int thread_index, int sample_index, float roughness, float eta, float3 omega_i) {
	float  rand_fresnel = random<SampleDimension::RUSSIAN_ROULETTE>(thread_index, 0, sample_index).y;
	float2 rand_brdf    = random<SampleDimension::BSDF_0>          (thread_index, 0, sample_index);

	float alpha_x = roughness;
	float alpha_y = roughness;

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

__device__ inline int lut_index(int i, int r, int c) {
	return i + r * LUT_DIM_IOR + c * LUT_DIM_IOR * LUT_DIM_ROUGHNESS;
}

__device__ inline float map_ior(int index_ior) {
#if 1
	return remap((float(index_ior) + 0.5f) / float(LUT_DIM_IOR), 0.0f, 1.0f, LUT_MIN_IOR, LUT_MAX_IOR);
#else
	return 0.0001f + remap((float(index_ior)) / float(LUT_DIM_IOR-1), 0.0f, 1.0f, LUT_MIN_IOR, LUT_MAX_IOR);
#endif
}

__device__ inline float map_roughness(int index_roughness) {
#if 1
	return square((float(index_roughness) + 0.5f) / float(LUT_DIM_ROUGHNESS));
#else
	return fmaxf(1e-6f, square((float(index_roughness)) / float(LUT_DIM_ROUGHNESS-1)));
#endif
}

__device__ inline float map_cos_theta(int index_cos_theta) {
#if 1
	return (float(index_cos_theta) + 0.5f) / float(LUT_DIM_COS_THETA);
#else
	return clamp((float(index_cos_theta)) / float(LUT_DIM_COS_THETA-1), 0.001f, 0.999f);
#endif
}

extern "C" __global__ void kernel_bsdf_integrate(bool entering_material, float * lut_directional_albedo) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= LUT_DIM_IOR * LUT_DIM_COS_THETA * LUT_DIM_ROUGHNESS) return;

	int i = (thread_index)                                     % LUT_DIM_IOR;
	int r = (thread_index / (LUT_DIM_IOR))                     % LUT_DIM_ROUGHNESS;
	int c = (thread_index / (LUT_DIM_IOR * LUT_DIM_ROUGHNESS)) % LUT_DIM_COS_THETA;

	float ior = map_ior(i);
	float eta = entering_material ? 1.0f / ior : ior;

	float alpha = map_roughness(r);

	float cos_theta = map_cos_theta(c);
	float sin_theta = safe_sqrt(1.0f - square(cos_theta));
	float3 omega_i = make_float3(sin_theta, 0.0f, cos_theta);

	float avg = 0.0f;
	constexpr int NUM_SAMPLES = 100000;

	for (int s = 0; s < NUM_SAMPLES; s++) {
		float weight = sample_dielectric(thread_index, s, alpha, eta, omega_i);
		avg = online_average(avg, weight, s + 1);
	}

	lut_directional_albedo[thread_index] = avg;
}

extern "C" __global__ void kernel_bsdf_average(const float * lut_directional_albedo, float * lut_albedo) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (thread_index >= LUT_DIM_IOR * LUT_DIM_ROUGHNESS) return;

	int i = (thread_index)               % LUT_DIM_IOR;
	int r = (thread_index / LUT_DIM_IOR) % LUT_DIM_ROUGHNESS;

	float avg = 0.0f;

	for (int c = 0; c < LUT_DIM_COS_THETA; c++) {
		float cos_theta = map_cos_theta(c);

		int index = lut_index(i, r, c);
		avg = online_average(avg, lut_directional_albedo[index] * cos_theta, c + 1);
	}

	lut_albedo[thread_index] = 2.0f * avg;
}
