#pragma once
#include "BVH/BVH.h"
#include "BVH/Builders/BVHPartitions.h"

#include "Util/BitArray.h"

struct SBVHBuilder {
private:
	static constexpr int SBVH_OVERALLOCATION = 4; // SBVH requires more space

	BVH * sbvh = nullptr;

	using PrimitiveRef = BVHPartitions::PrimitiveRef;

	PrimitiveRef * indices_x = nullptr;
	PrimitiveRef * indices_y = nullptr;
	PrimitiveRef * indices_z = nullptr;

	PrimitiveRef * indices_going_right_x = nullptr;
	PrimitiveRef * indices_going_right_y = nullptr;
	PrimitiveRef * indices_going_right_z = nullptr;
	int            indices_going_right_offset;

	// Scatch memory
	float * sah    = nullptr;
	AABB  * bounds = nullptr;
	BitArray indices_going_left;
	BitArray indices_going_right;

	int max_primitives_in_leaf;

	int SBVHBuilder::build_sbvh(BVHNode & node, const Triangle * triangles, PrimitiveRef * indices[3], int & node_index, int first_index, int index_count, float inv_root_surface_area);

public:
	inline void init(BVH * sbvh, int triangle_count, int max_primitives_in_leaf) {
		this->sbvh = sbvh;
		this->max_primitives_in_leaf = max_primitives_in_leaf;

		PrimitiveRef * indices_xyz = new PrimitiveRef[3 * SBVH_OVERALLOCATION * triangle_count];
		indices_x = indices_xyz;
		indices_y = indices_xyz + SBVH_OVERALLOCATION * triangle_count;
		indices_z = indices_xyz + SBVH_OVERALLOCATION * triangle_count * 2;

		PrimitiveRef * indices_going_right_xyz = new PrimitiveRef[3 * 2 * triangle_count];
		indices_going_right_x = indices_going_right_xyz;
		indices_going_right_y = indices_going_right_xyz + 2 * triangle_count;
		indices_going_right_z = indices_going_right_xyz + 2 * triangle_count * 2;

		sah    = new float[triangle_count];
		bounds = new AABB [triangle_count * 2 + 1];
		indices_going_left .init(triangle_count);
		indices_going_right.init(triangle_count);

		sbvh->indices = new int    [SBVH_OVERALLOCATION * triangle_count];
		sbvh->nodes   = new BVHNode[SBVH_OVERALLOCATION * triangle_count];
	}

	inline void free() {
		delete [] indices_x;
		delete [] indices_going_right_x;

		delete [] sah;
		delete [] bounds;
		indices_going_left .free();
		indices_going_right.free();
	}

	void build(const Triangle * triangles, int triangle_count); // SAH-based object + spatial splits, Stich et al. 2009 (Triangles only)
};
