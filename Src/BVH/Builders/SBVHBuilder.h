#pragma once
#include "BVH/BVH.h"

#include "Util/Array.h"
#include "Util/BitArray.h"

struct PrimitiveRef;

struct SBVHBuilder {
private:
	static constexpr int SBVH_OVERALLOCATION = 4; // SBVH requires more space

	BVH * sbvh = nullptr;

	PrimitiveRef * indices_x = nullptr;
	PrimitiveRef * indices_y = nullptr;
	PrimitiveRef * indices_z = nullptr;

	PrimitiveRef * indices_going_right_x = nullptr;
	PrimitiveRef * indices_going_right_y = nullptr;
	PrimitiveRef * indices_going_right_z = nullptr;
	int            indices_going_right_offset;

	// Scatch memory
	float * sah = nullptr;
	BitArray indices_going_left;
	BitArray indices_going_right;

	int build_sbvh(BVHNode2 & node, const Array<Triangle> & triangles, PrimitiveRef * indices[3], int & node_index, int first_index, int index_count, float inv_root_surface_area);

public:
	void init(BVH * sbvh, int triangle_count);
	void free();

	void build(const Array<Triangle> & triangles); // SAH-based object + spatial splits, Stich et al. 2009 (Triangles only)
};
