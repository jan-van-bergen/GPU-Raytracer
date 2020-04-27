#pragma once
#include <fstream>
#include <ostream>
#include <algorithm>

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

namespace BVHBuilders {
	void build_bvh(BVHNode & node, const Triangle * triangles, int * indices[3], BVHNode nodes[], int & node_index, int first_index, int index_count, float * sah, int * temp);
	int build_sbvh(BVHNode & node, const Triangle * triangles, int * indices[3], BVHNode nodes[], int & node_index, int first_index, int index_count, float * sah, int * temp[2], float inv_root_surface_area, AABB node_aabb);
}

struct BVH {
	int        triangle_count;
	Triangle * triangles;

	int * indices;

	int       node_count;
	BVHNode * nodes;

	int leaf_count;

	inline void init(int count) {
		assert(count > 0);

		triangle_count = count; 
		triangles = new Triangle[triangle_count];

		// Construct Node pool
		nodes = reinterpret_cast<BVHNode *>(ALLIGNED_MALLOC(2 * triangle_count * sizeof(BVHNode), 64));
		assert((unsigned long long)nodes % 64 == 0);
	}

	inline void build_bvh() {
		// Construct index arrays for all three dimensions
		int * indices_x = new int[triangle_count];
		int * indices_y = new int[triangle_count];
		int * indices_z = new int[triangle_count];

		for (int i = 0; i < triangle_count; i++) {
			indices_x[i] = i;
			indices_y[i] = i;
			indices_z[i] = i;
		}

		std::sort(indices_x, indices_x + triangle_count, [&](int a, int b) { return triangles[a].get_position().x < triangles[b].get_position().x; });
		std::sort(indices_y, indices_y + triangle_count, [&](int a, int b) { return triangles[a].get_position().y < triangles[b].get_position().y; });
		std::sort(indices_z, indices_z + triangle_count, [&](int a, int b) { return triangles[a].get_position().z < triangles[b].get_position().z; });
		
		int * indices_3[3] = { indices_x, indices_y, indices_z };
		
		float * sah = new float[triangle_count];

		int * temp = new int[triangle_count];

		int node_index = 2;
		BVHBuilders::build_bvh(nodes[0], triangles, indices_3, nodes, node_index, 0, triangle_count, sah, temp);

		indices = indices_x;
		delete [] indices_y;
		delete [] indices_z;

		assert(node_index <= 2 * triangle_count);

		node_count = node_index;
		leaf_count = triangle_count;

		delete [] temp;
		delete [] sah;
	}

	inline void build_sbvh() {
		const int overallocation = 2; // SBVH requires more space

		int * indices_x = new int[overallocation * triangle_count];
		int * indices_y = new int[overallocation * triangle_count];
		int * indices_z = new int[overallocation * triangle_count];

		for (int i = 0; i < triangle_count; i++) {
			indices_x[i] = i;
			indices_y[i] = i;
			indices_z[i] = i;
		}

		std::sort(indices_x, indices_x + triangle_count, [&](int a, int b) { return triangles[a].get_position().x < triangles[b].get_position().x; });
		std::sort(indices_y, indices_y + triangle_count, [&](int a, int b) { return triangles[a].get_position().y < triangles[b].get_position().y; });
		std::sort(indices_z, indices_z + triangle_count, [&](int a, int b) { return triangles[a].get_position().z < triangles[b].get_position().z; });
		
		int * indices_3[3] = { indices_x, indices_y, indices_z };
		
		float * sah = new float[triangle_count];
		
		int * temp[2] = { new int[triangle_count], new int[triangle_count] };

		AABB root_aabb = BVHPartitions::calculate_bounds(triangles, indices_3[0], 0, triangle_count);

		int node_index = 2;
		leaf_count = BVHBuilders::build_sbvh(nodes[0], triangles, indices_3, nodes, node_index, 0, triangle_count, sah, temp, 1.0f / root_aabb.surface_area(), root_aabb);
		
		indices = indices_x;
		delete [] indices_y;
		delete [] indices_z;

		printf("SBVH Leaf count: %i\n", leaf_count);

		assert(node_index <= 2 * triangle_count);

		node_count = node_index;

		delete [] temp[0];
		delete [] temp[1];
		delete [] sah;
	}

	inline void save_to_disk(const char * bvh_filename) const {
		FILE * file;
		fopen_s(&file, bvh_filename, "wb");

		if (file == nullptr) abort();

		fwrite(reinterpret_cast<const char *>(&triangle_count), sizeof(int), 1, file);
		fwrite(reinterpret_cast<const char *>(triangles), sizeof(Triangle), triangle_count, file);

		fwrite(reinterpret_cast<const char *>(&node_count), sizeof(int), 1, file);
		fwrite(reinterpret_cast<const char *>(nodes), sizeof(BVHNode), node_count, file);

		fwrite(reinterpret_cast<const char *>(&leaf_count), sizeof(int), 1, file);
		
		fwrite(reinterpret_cast<const char *>(indices), sizeof(int), leaf_count, file);

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

		fread(reinterpret_cast<char *>(&leaf_count), sizeof(int), 1, file);
			
		indices = new int[leaf_count];
		fread(reinterpret_cast<char *>(indices), sizeof(int), leaf_count, file);

		fclose(file);
	}
};
