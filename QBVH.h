#pragma once
#include "BVH.h"
#include "BVHBuilders.h"

struct QBVHNode {
	float aabb_min_x[4] = { 0.0f };
	float aabb_min_y[4] = { 0.0f };
	float aabb_min_z[4] = { 0.0f };
	float aabb_max_x[4] = { 0.0f };
	float aabb_max_y[4] = { 0.0f };
	float aabb_max_z[4] = { 0.0f };
	
	struct {
		int index;
		int count;
	} index_and_count[4];

	inline       int & get_index(int i)       { return index_and_count[i].index; }
	inline const int & get_index(int i) const { return index_and_count[i].index; }
	inline       int & get_count(int i)       { return index_and_count[i].count; }
	inline const int & get_count(int i) const { return index_and_count[i].count; }

	inline int get_child_count() const {
		int result = 0;

		for (int i = 0; i < 4; i++) {
			if (get_count(i) == -1) break;

			result++;
		}

		return result;
	}

};

static_assert(sizeof(QBVHNode) == 128);

struct QBVH {
	int        triangle_count;
	Triangle * triangles;

	int   index_count;
	int * indices;

	int        node_count;
	QBVHNode * nodes;
};
