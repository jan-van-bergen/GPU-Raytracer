#pragma once
#include "Math/Vector3.h"

#include "Texture.h"

#include "../CUDA_Source/Common.h"

struct Material {
	const char * name;

	enum struct Type : char {
		LIGHT      = 0,
		DIFFUSE    = 1,
		DIELECTRIC = 2,
		GLOSSY     = 3
	};

	Type type = Type::DIFFUSE;

	Vector3 emission;

	Vector3 diffuse = Vector3(1.0f, 1.0f, 1.0f);
	TextureHandle texture_id;

	Vector3 transmittance = Vector3(0.0f);
	float index_of_refraction = 1.0f;

	float linear_roughness = 0.5f;
};

struct MaterialHandle {
	int handle = INVALID;

	static inline MaterialHandle get_default() { return MaterialHandle { 0 }; }
};
