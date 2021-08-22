#pragma once
#include "Math/AABB.h"
#include "Math/Vector2.h"
#include "Math/Vector3.h"

struct Triangle {
	AABB aabb;

	Vector3 position_0;
	Vector3 position_1;
	Vector3 position_2;

	Vector3 normal_0;
	Vector3 normal_1;
	Vector3 normal_2;

	Vector2 tex_coord_0;
	Vector2 tex_coord_1;
	Vector2 tex_coord_2;

	inline void init() {
		// Check wheter any normal is invalid, if so replace it with the geometric normal of this Triangle
		bool normal_0_invalid = Math::approx_equal(Vector3::length(normal_0), 0.0f);
		bool normal_1_invalid = Math::approx_equal(Vector3::length(normal_1), 0.0f);
		bool normal_2_invalid = Math::approx_equal(Vector3::length(normal_2), 0.0f);

		if (normal_0_invalid || normal_1_invalid || normal_2_invalid) {
			Vector3 geometric_normal = Vector3::normalize(Vector3::cross(
				position_1 - position_0,
				position_2 - position_0
			));
			if (normal_0_invalid) normal_0 = geometric_normal;
			if (normal_1_invalid) normal_1 = geometric_normal;
			if (normal_2_invalid) normal_2 = geometric_normal;
		}

		// Calc AABB from vertices
		Vector3 vertices[3] = { position_0, position_1, position_2 };
		aabb = AABB::from_points(vertices, 3);
	}

	inline Vector3 get_center() const {
		return (position_0 + position_1 + position_2) / 3.0f;
	}
};
