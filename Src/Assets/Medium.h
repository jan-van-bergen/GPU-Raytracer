#pragma once
#include "Math/Math.h"
#include "Math/Vector3.h"

#include "../CUDA_Source/Common.h"

struct Medium {
	const char * name;

	// Artist friendly parameters as described in
	// Chiang et al. - Practical and Controllable Subsurface Scattering for Production Path Tracing
	Vector3 A; // Multi-scatter albedo
	Vector3 d; // Scattering distance

	float g = 0.0f;

	static float A_to_alpha(float A) {
		return 1.0f - expf(-5.09406f*A + 2.61188f*A*A - 4.31805f*A*A*A);
	}

	static float A_to_s(float A) {
		return 1.9f - A + 3.5f * Math::square(A - 0.8f);
	}

	void set_A_and_d(const Vector3 & sigma_a, const Vector3 & sigma_s) {
		Vector3 sigma_t = sigma_a + sigma_s;

		if (sigma_t.x == 0.0f && sigma_t.y == 0.0f && sigma_t.z == 0.0f) {
			A = Vector3(0.0f);
			d = Vector3(100000.0f);
			return;
		}

		Vector3 alpha = sigma_s / sigma_t; // Single scatter albedo
		A.x = Math::invert_monotonically_increasing_function(alpha.x, A_to_alpha);
		A.y = Math::invert_monotonically_increasing_function(alpha.y, A_to_alpha);
		A.z = Math::invert_monotonically_increasing_function(alpha.z, A_to_alpha);

		Vector3 s = Vector3(A_to_s(A.x), A_to_s(A.y), A_to_s(A.z));
		d = 1.0f / (sigma_t * s);
	}

	void get_sigmas(Vector3 & sigma_a, Vector3 & sigma_s) const {
		Vector3 alpha = Vector3(A_to_alpha(A.x), A_to_alpha(A.y), A_to_alpha(A.z));
		Vector3 s     = Vector3(A_to_s    (A.x), A_to_s    (A.y), A_to_s    (A.z));

		Vector3 sigma_t = 1.0f / (Vector3::max(d, 0.0001f) * s);

		sigma_s = alpha * sigma_t;
		sigma_a = sigma_t - sigma_s;
	}
};

struct MediumHandle { int handle = INVALID; };
