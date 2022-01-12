#pragma once
#include "BVH/BVH.h"
#include "BVHPartitions.h"

#include "Core/Array.h"
#include "Core/BitArray.h"

struct PrimitiveRef;

struct SBVHBuilder {

	BVH2 * sbvh = nullptr;

	Array<PrimitiveRef> indices[3];

	// Scatch memory
	Array<float> sah;
	BitArray indices_going_left;

	float inv_root_surface_area;

	SBVHBuilder(BVH2 * sbvh, size_t triangle_count) : sbvh(sbvh), sah(triangle_count), indices_going_left(triangle_count) { }

	void build(const Array<Triangle> & triangles); // SAH-based object + spatial splits, Stich et al. 2009 (Triangles only)

private:
	int build_sbvh(BVHNode2 & node, const Array<Triangle> & triangles, int first_index, int index_count);
};
