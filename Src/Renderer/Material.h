#pragma once
#include "Core/String.h"

#include "Math/Vector3.h"

#include "Handle.h"

#include "Medium.h"
#include "Texture.h"

struct Material {
	String name;

	enum struct Type : char {
		LIGHT,
		DIFFUSE,
		PLASTIC,
		DIELECTRIC,
		CONDUCTOR
	};

	Type type = Type::DIFFUSE;

	Vector3 emission;

	Vector3         diffuse = Vector3(1.0f, 1.0f, 1.0f);
	Handle<Texture> texture_handle;

	Handle<Medium> medium_handle;
	float          index_of_refraction = 1.33f;

	Vector3 eta = Vector3(1.33f);
	Vector3 k   = Vector3(1.0f);

	float linear_roughness = 0.5f;

	bool is_light() const {
		return type == Type::LIGHT && Vector3::length_squared(emission) > 0.0f;
	}
};
