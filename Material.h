#pragma once
#include "Texture.h"

struct Material {
	enum Type : char {
		DIFFUSE    = 0,
		DIELECTRIC = 1,
		GLOSSY     = 2
	};

	Type type = DIFFUSE;

	Vector3 diffuse = Vector3(1.0f, 1.0f, 1.0f);	
	int texture_id = -1;

	Vector3 emittance;

	float index_of_refraction = 1.0f;

	float alpha = roughness_to_alpha(0.1f);

	// Based on PBRT
	inline static float roughness_to_alpha(float roughness) {
		const float limit = 1e-3f;
        if (roughness < limit) {
			roughness = limit;
		}

        float x = log(roughness);
        return 1.62142f
            + 0.819955f    * x
            + 0.1734f      * x * x
            + 0.0171201f   * x * x * x
            + 0.000640711f * x * x * x * x;
	}
};
