#pragma once
#include <stdio.h>

#include "Matrix4.h"

struct AABB {
	Vector3 min;
	Vector3 max;
	
	static AABB create_empty();

	static AABB from_points(const Vector3 * points, int point_count);

	inline bool is_valid() const {
		return max.x >= min.x && max.y >= min.y && max.z >= min.z;
	}

	inline bool is_empty() const {
		return 
			min.x ==  INFINITY && min.y ==  INFINITY && min.z ==  INFINITY &&
			max.x == -INFINITY && max.y == -INFINITY && max.z == -INFINITY;
	}

	// Make sure the AABB is non-zero along every dimension
	inline void fix_if_needed(float epsilon = 0.001f) {
		for (int dimension = 0; dimension < 3; dimension++) {
			if (max[dimension] - min[dimension] < epsilon) {
				min[dimension] -= 0.5f * epsilon;
				max[dimension] += 0.5f * epsilon;
			}
		}
	}

	inline float surface_area() const {
		assert(is_valid() || is_empty());

		Vector3 diff = max - min;

		return 2.0f * (diff.x * diff.y + diff.y * diff.z + diff.z * diff.x);
	}

	inline void expand(const Vector3 & point) {
		min = Vector3::min(min, point);
		max = Vector3::max(max, point);
	}

	inline void expand(const AABB & aabb) {
		min = Vector3::min(min, aabb.min);
		max = Vector3::max(max, aabb.max);
	}

	inline Vector3 get_center() const {
		return (min + max) * 0.5f;
	}

	inline void debug(FILE * file, int index) const {
		Vector3 vertices[8] = {
			Vector3(min.x, min.y, min.z),
			Vector3(min.x, min.y, max.z),
			Vector3(max.x, min.y, max.z),
			Vector3(max.x, min.y, min.z),
			Vector3(min.x, max.y, min.z),
			Vector3(min.x, max.y, max.z),
			Vector3(max.x, max.y, max.z),
			Vector3(max.x, max.y, min.z)
		};

		int faces[36] = {
			1, 2, 3, 1, 3, 4,
			1, 2, 6, 1, 6, 5,
			1, 5, 8, 1, 8, 4,
			4, 8, 7, 4, 7, 3,
			3, 7, 6, 3, 6, 2,
			5, 6, 7, 5, 7, 8
		};

		for (int v = 0; v < 8; v++) {
			fprintf_s(file, "v %f %f %f\n", vertices[v].x, vertices[v].y, vertices[v].z);
		}

		for (int f = 0; f < 36; f += 3) {
			fprintf_s(file, "f %i %i %i\n", 8*index + faces[f], 8*index + faces[f+1], 8*index + faces[f+2]);
		}
	}

	static AABB unify  (const AABB & b1, const AABB & b2);
	static AABB overlap(const AABB & b1, const AABB & b2);
	
	static AABB transform(const AABB & aabb, const Matrix4 & transformation);
};
