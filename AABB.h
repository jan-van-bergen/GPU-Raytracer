#pragma once
#include "Matrix4.h"

struct AABB {
	Vector3 min;
	Vector3 max;
	
	static AABB create_empty();

	static AABB from_points(const Vector3 * points, int point_count);

	inline bool is_valid() const {
		return max.x > min.x && max.y > min.y && max.z > min.z;
	}

	inline bool is_empty() const {
		return 
			min.x ==  INFINITY && min.y ==  INFINITY && min.z ==  INFINITY &&
			max.x == -INFINITY && max.y == -INFINITY && max.z == -INFINITY;
	}

	// Make sure the AABB is non-zero along every dimension
	inline void fix_if_needed() {
		for (int dimension = 0; dimension < 3; dimension++) {
			if (max[dimension] - min[dimension] < 0.001f) {
				max[dimension] += 0.005f;
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

	static AABB overlap(const AABB & b1, const AABB & b2);
	
	static AABB transform(const AABB & aabb, const Matrix4 & transformation);
};
