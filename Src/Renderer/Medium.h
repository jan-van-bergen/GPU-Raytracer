#pragma once
#include "Core/String.h"

#include "Math/Math.h"
#include "Math/Vector3.h"

#include "CUDA/Common.h"

struct Medium {
	String name;

	Vector3 C   = 1.0f; // Multi-scatter albedo
	Vector3 mfp = 1.0f; // Mean free path

	float g = 0.0f; // Henyey-Greenstein mean cosine

	void from_sigmas(const Vector3 & sigma_a, const Vector3 & sigma_s) {
		Vector3 sigma_t = sigma_a + sigma_s;

		// Van de Hulst albedo inversion
		Vector3 alpha = sigma_s / sigma_t; // Single scatter albedo
		Vector3 s = Vector3::apply((1.0f - alpha) / (1.0f - alpha*g), sqrtf);

		C   = (1.0f - s)*(1.0f - 0.139f*s) / (1.0f + 1.17f*s);
		mfp = 1.0f / sigma_t;
	}

	void to_sigmas(Vector3 & sigma_a, Vector3 & sigma_s) const {
		// Van de Hulst albedo inversion
		Vector3 s = 4.09712f + 4.20863f*C - Vector3::apply(9.59217f + 41.6808f*C + 17.7126f*C*C, sqrtf);
		Vector3 alpha = (1.0f - s*s) / (1.0f - Math::clamp(g, -0.999f, 0.999f)*s*s);

		Vector3 sigma_t = 1.0f / Vector3::max(mfp, 1e-6f);
		sigma_s = alpha * sigma_t;
		sigma_a = sigma_t - sigma_s;
	}
};
