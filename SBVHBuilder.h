#pragma once
#include "BVH.h"

struct SBVHBuilder {
private:
	static constexpr int SBVH_OVERALLOCATION = 4; // SBVH requires more space

	BVH * sbvh = nullptr;
	
	int * indices_x = nullptr;
	int * indices_y = nullptr;
	int * indices_z = nullptr;
	
	float * sah     = nullptr;
	int   * temp[2] = { };

public:
	inline void init(BVH * sbvh, int triangle_count) {
		this->sbvh = sbvh;

		indices_x = new int[SBVH_OVERALLOCATION * triangle_count];
		indices_y = new int[SBVH_OVERALLOCATION * triangle_count];
		indices_z = new int[SBVH_OVERALLOCATION * triangle_count];

		for (int i = 0; i < triangle_count; i++) {
			indices_x[i] = i;
			indices_y[i] = i;
			indices_z[i] = i;
		}

		sah     = new float[triangle_count];
		temp[0] = new int  [triangle_count];
		temp[1] = new int  [triangle_count];
		
		sbvh->indices = indices_x;
		sbvh->nodes   = new BVHNode[SBVH_OVERALLOCATION * triangle_count];
	}

	inline void free() {
		delete [] indices_y;
		delete [] indices_z;

		delete [] sah;
		delete [] temp[0];
		delete [] temp[1];
	}

	void build(const Triangle * triangles, int triangle_count); // SAH-based object + spatial splits, Stich et al. 2009 (Triangles only)
};
