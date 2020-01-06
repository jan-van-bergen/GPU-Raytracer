#include "AABB.h"

AABB AABB::create_empty() {
	AABB aabb;
	aabb.min = Vector3(+INFINITY);
	aabb.max = Vector3(-INFINITY);

	return aabb;
}

AABB AABB::from_points(const Vector3 * points, int point_count) {
	AABB aabb = create_empty();

	for (int i = 0; i < point_count; i++) {
		aabb.expand(points[i]);
	}

	aabb.fix_if_needed();
	assert(aabb.is_valid());

	return aabb;
}

AABB AABB::overlap(const AABB & b1, const AABB & b2) {
	assert(b1.is_valid() || b1.is_empty());
	assert(b2.is_valid() || b2.is_empty());

	AABB aabb;
	aabb.min = Vector3::max(b1.min, b2.min);
	aabb.max = Vector3::min(b1.max, b2.max);

	if (!aabb.is_valid()) aabb = create_empty();

	return aabb;
}

AABB AABB::transform(const AABB & aabb, const Matrix4 & transformation) {
	Vector3 corners[8] = {
		Vector3(aabb.min.x, aabb.min.y, aabb.min.z),
		Vector3(aabb.min.x, aabb.min.y, aabb.max.z),
		Vector3(aabb.max.x, aabb.min.y, aabb.max.z),
		Vector3(aabb.max.x, aabb.min.y, aabb.min.z),
		Vector3(aabb.min.x, aabb.max.y, aabb.min.z),
		Vector3(aabb.min.x, aabb.max.y, aabb.max.z),
		Vector3(aabb.max.x, aabb.max.y, aabb.max.z),
		Vector3(aabb.max.x, aabb.max.y, aabb.min.z)
	};

	for (int i = 0; i < 8; i++) {
		corners[i] = Matrix4::transform_position(transformation, corners[i]);
	}

	return AABB::from_points(corners, 8);
}
