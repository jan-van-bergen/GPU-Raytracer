#pragma once
#include <fstream>
#include <ostream>
#include <algorithm>

#include "BVHBuilders.h"
#include "BVHPartitions.h"

struct BVHNode {
	AABB aabb;
	union {
		int left;
		int first;
	};
	int count; // Stores split axis in its 2 highest bits, count in its lowest 30 bits

	inline int get_count() const {
		return count & ~BVH_AXIS_MASK;
	}

	inline bool is_leaf() const {
		return get_count() > 0;
	}
};

struct BVH {
	int        triangle_count;
	Triangle * triangles;

	int   index_count;
	int * indices;

	int       node_count;
	BVHNode * nodes;

	inline void save_to_disk(const char * bvh_filename) const {
		FILE * file;
		fopen_s(&file, bvh_filename, "wb");

		if (file == nullptr) abort();

		fwrite(reinterpret_cast<const char *>(&triangle_count), sizeof(int), 1, file);
		fwrite(reinterpret_cast<const char *>(triangles), sizeof(Triangle), triangle_count, file);

		fwrite(reinterpret_cast<const char *>(&node_count), sizeof(int), 1, file);
		fwrite(reinterpret_cast<const char *>(nodes), sizeof(BVHNode), node_count, file);

		fwrite(reinterpret_cast<const char *>(&index_count), sizeof(int), 1, file);
		fwrite(reinterpret_cast<const char *>(indices), sizeof(int), index_count, file);

		fclose(file);
	}

	inline void load_from_disk(const char * bvh_filename) {
		FILE * file;
		fopen_s(&file, bvh_filename, "rb"); 
		
		if (file == nullptr) abort();

		fread(reinterpret_cast<char *>(&triangle_count), sizeof(int), 1, file);

		triangles = new Triangle[triangle_count];
		fread(reinterpret_cast<char *>(triangles), sizeof(Triangle), triangle_count, file);
		
		fread(reinterpret_cast<char *>(&node_count), sizeof(int), 1, file);

		nodes = new BVHNode[node_count];
		fread(reinterpret_cast<char *>(nodes), sizeof(BVHNode), node_count, file);

		fread(reinterpret_cast<char *>(&index_count), sizeof(int), 1, file);
			
		indices = new int[index_count];
		fread(reinterpret_cast<char *>(indices), sizeof(int), index_count, file);

		fclose(file);
	}
};
