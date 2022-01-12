#pragma once
#include "BVH/BVH.h"
#include "BVHPartitions.h"

#include "Util/Array.h"
#include "Util/BitArray.h"

struct PrimitiveRef;

struct SBVHBuilder {
private:
	BVH2 * sbvh = nullptr;

	Array<PrimitiveRef> indices[3];

	// Scatch memory
	Array<float> sah;
	BitArray indices_going_left;

	float inv_root_surface_area;

	int build_sbvh(BVHNode2 & node, const Array<Triangle> & triangles, int first_index, int index_count);

public:
	void init(BVH2 * sbvh, int triangle_count);
	void free();

	void build(const Array<Triangle> & triangles); // SAH-based object + spatial splits, Stich et al. 2009 (Triangles only)
};
