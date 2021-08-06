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

	inline void calc_aabb() {
		Vector3 vertices[3] = { position_0, position_1, position_2 };
		aabb = AABB::from_points(vertices, 3);
	}

	inline Vector3 get_center() const {
		return (position_0 + position_1 + position_2) / 3.0f;
	}
};
