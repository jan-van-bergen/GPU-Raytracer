#pragma once
#include "BVH.h"
#include "BVHPartitions.h"

struct SBVHBuilder {
private:
	static constexpr int SBVH_OVERALLOCATION = 4; // SBVH requires more space

	BVH * sbvh = nullptr;
	
	using PrimitiveRef = BVHPartitions::PrimitiveRef;

	PrimitiveRef * indices_x = nullptr;
	PrimitiveRef * indices_y = nullptr;
	PrimitiveRef * indices_z = nullptr;
	
	float * sah     = nullptr;
	int   * temp[2] = { };

	int max_primitives_in_leaf;

	int SBVHBuilder::build_sbvh(BVHNode & node, const Triangle * triangles, PrimitiveRef * indices[3], int & node_index, int first_index, int index_count, float inv_root_surface_area);

public:
	inline void init(BVH * sbvh, int triangle_count, int max_primitives_in_leaf) {
		this->sbvh = sbvh;
		this->max_primitives_in_leaf = max_primitives_in_leaf;

		indices_x = new PrimitiveRef[SBVH_OVERALLOCATION * triangle_count];
		indices_y = new PrimitiveRef[SBVH_OVERALLOCATION * triangle_count];
		indices_z = new PrimitiveRef[SBVH_OVERALLOCATION * triangle_count];

		sah     = new float[triangle_count];
		temp[0] = new int  [triangle_count];
		temp[1] = new int  [triangle_count];
		
		sbvh->indices = new int    [SBVH_OVERALLOCATION * triangle_count];
		sbvh->nodes   = new BVHNode[SBVH_OVERALLOCATION * triangle_count];
	}

	inline void free() {
		delete [] indices_x;
		delete [] indices_y;
		delete [] indices_z;

		delete [] sah;
		delete [] temp[0];
		delete [] temp[1];
	}

	void build(const Triangle * triangles, int triangle_count); // SAH-based object + spatial splits, Stich et al. 2009 (Triangles only)
};
