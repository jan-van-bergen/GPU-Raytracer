#pragma once
#include "Texture.h"

struct Material {
	enum Type : char {
		LIGHT      = 0,
		DIFFUSE    = 1,
		DIELECTRIC = 2,
		GLOSSY     = 3
	};

	Type type = DIFFUSE;

	Vector3 diffuse = Vector3(1.0f, 1.0f, 1.0f);	
	int texture_id = -1;

	Vector3 emission;

	float index_of_refraction = 1.0f;

	float roughness = 0.5f;
};
