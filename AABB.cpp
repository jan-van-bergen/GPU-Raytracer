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

// Based on: https://zeux.io/2010/10/17/aabb-from-obb-with-component-wise-abs/
AABB AABB::transform(const AABB & aabb, const Matrix4 & transformation) {
	Vector3 center = 0.5f * (aabb.min + aabb.max);
	Vector3 extent = 0.5f * (aabb.max - aabb.min);

	Vector3 new_center = Matrix4::transform_position (             transformation,  center);
	Vector3 new_extent = Matrix4::transform_direction(Matrix4::abs(transformation), extent);

	AABB result;
	result.min = new_center - new_extent;
	result.max = new_center + new_extent;

	return result;
}
