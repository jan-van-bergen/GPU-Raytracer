#pragma once
#include "Core/Array.h"

#include "Math/Math.h"
#include "Math/AABB.h"

#include "Util/Util.h"

struct Triangle;
struct Mesh;

struct PrimitiveRef {
	int  index;
	AABB aabb;
};

struct ObjectSplit {
	int   index;
	float cost;
	int   dimension;

	AABB aabb_left;
	AABB aabb_right;
};

struct SpatialSplit {
	int   index;
	float cost;
	int   dimension;

	AABB aabb_left;
	AABB aabb_right;

	float plane_distance;

	int num_left;
	int num_right;
};

// Contains various ways to parition space into "left" and "right" as well as helper methods
namespace BVHPartitions {
	inline constexpr int SBVH_BIN_COUNT = 256;

	ObjectSplit partition_sah(const Array<Triangle> & triangles, int * indices[3], int first_index, int index_count, float * sah);
	ObjectSplit partition_sah(const Array<Mesh>     & meshes,    int * indices[3], int first_index, int index_count, float * sah);

	ObjectSplit partition_sah(Array<PrimitiveRef> primitive_refs[3], int first_index, int index_count, float * sah);

	void triangle_intersect_plane(Vector3 vertices[3], int dimension, float plane, Vector3 intersections[], int * intersection_count);

	SpatialSplit partition_spatial(const Array<Triangle> & triangles, const Array<PrimitiveRef> indices[3], int first_index, int index_count, float * sah, AABB bounds);
}
