#pragma once
#include "Math/Vector3.h"

#include "Medium.h"
#include "Texture.h"

struct Material {
	const char * name;

	enum struct Type : char {
		LIGHT,
		DIFFUSE,
		PLASTIC,
		DIELECTRIC,
		CONDUCTOR
	};

	Type type = Type::DIFFUSE;

	Vector3 emission;

	Vector3       diffuse = Vector3(1.0f, 1.0f, 1.0f);
	TextureHandle texture_id;

	MediumHandle medium_handle;
	float        index_of_refraction = 1.33f;

	Vector3 eta = Vector3(1.33f);
	Vector3 k   = Vector3(1.0f);

	float linear_roughness = 0.5f;
};

struct MaterialHandle {
	int handle = INVALID;

	static inline MaterialHandle get_default() { return MaterialHandle { 0 }; }
};
