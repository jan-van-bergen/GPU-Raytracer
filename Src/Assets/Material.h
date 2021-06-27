#pragma once

#include "Math/Vector3.h"

struct Material {
	const char * name;

	enum class Type : char {
		LIGHT      = 0,
		DIFFUSE    = 1,
		DIELECTRIC = 2,
		GLOSSY     = 3
	};

	Type type = Type::DIFFUSE;
	
	Vector3 emission;

	Vector3 diffuse = Vector3(1.0f, 1.0f, 1.0f);	
	int texture_id = -1;

	Vector3 transmittance = Vector3(0.0f);
	float index_of_refraction = 1.0f;

	float linear_roughness = 0.5f;
};
