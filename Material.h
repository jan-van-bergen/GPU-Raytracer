#pragma once
#include "Texture.h"

struct Material {
	enum Type : char {
		DIFFUSE    = 0,
		DIELECTRIC = 1
	};

	Type type = DIFFUSE;

	Vector3 diffuse = Vector3(1.0f, 1.0f, 1.0f);	
	int texture_id = -1;

	Vector3 emittance;

	float index_of_refraction = 1.0f;

	/*Vector3 reflection;

	Vector3 transmittance;
	*/
};
