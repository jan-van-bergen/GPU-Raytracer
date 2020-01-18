#pragma once
#include <fstream>
#include <ostream>
#include <algorithm>

#include "BVHBuilders.h"

struct BVHNode {
	AABB aabb;
	union {
		int left;
		int first;
	};
	int count; // Stores split axis in its 2 highest bits, count in its lowest 30 bits
};

template<typename PrimitiveType>
struct BVH {
	int             primitive_count;
	PrimitiveType * primitives;

	int * indices_x;
	int * indices_y;
	int * indices_z;

	int       node_count;
	BVHNode * nodes;

	inline void init(int count) {
		assert(count > 0);

		primitive_count = count; 
		primitives = new PrimitiveType[primitive_count];

		int overallocation = 2; // SBVH requires more space

		// Construct index array
		int * all_indices  = new int[3 * overallocation * primitive_count];
		indices_x = all_indices;
		indices_y = all_indices + primitive_count * overallocation;
		indices_z = all_indices + primitive_count * overallocation * 2;

		for (int i = 0; i < primitive_count; i++) {
			indices_x[i] = i;
			indices_y[i] = i;
			indices_z[i] = i;
		}

		// Construct Node pool
		nodes = reinterpret_cast<BVHNode *>(ALLIGNED_MALLOC(2 * primitive_count * sizeof(BVHNode), 64));
		assert((unsigned long long)nodes % 64 == 0);
	}

	inline void build_bvh() {
		float * sah = new float[primitive_count];
		
		std::sort(indices_x, indices_x + primitive_count, [&](int a, int b) { return primitives[a].get_position().x < primitives[b].get_position().x; });
		std::sort(indices_y, indices_y + primitive_count, [&](int a, int b) { return primitives[a].get_position().y < primitives[b].get_position().y; });
		std::sort(indices_z, indices_z + primitive_count, [&](int a, int b) { return primitives[a].get_position().z < primitives[b].get_position().z; });
		
		int * indices[3] = { indices_x, indices_y, indices_z };

		int * temp = new int[primitive_count];

		int node_index = 2;
		BVHBuilders::build_bvh(nodes[0], primitives, indices, nodes, node_index, 0, primitive_count, sah, temp);

		assert(node_index <= 2 * primitive_count);

		node_count = node_index;

		delete [] temp;
		delete [] sah;
		
		PrimitiveType * new_primitives = new PrimitiveType[primitive_count];

		for (int i = 0; i < primitive_count; i++) {
			new_primitives[i] = primitives[indices_x[i]];
		}

		delete [] primitives;
		primitives = new_primitives;
	}

	inline void build_sbvh() {
		float * sah = new float[primitive_count];
		
		std::sort(indices_x, indices_x + primitive_count, [&](int a, int b) { return primitives[a].get_position().x < primitives[b].get_position().x; });
		std::sort(indices_y, indices_y + primitive_count, [&](int a, int b) { return primitives[a].get_position().y < primitives[b].get_position().y; });
		std::sort(indices_z, indices_z + primitive_count, [&](int a, int b) { return primitives[a].get_position().z < primitives[b].get_position().z; });
		
		int * indices[3] = { indices_x, indices_y, indices_z };

		int * temp[2] = { new int[primitive_count], new int[primitive_count] };

		AABB root_aabb = BVHPartitions::calculate_bounds(primitives, indices[0], 0, primitive_count);

		int node_index = 2;
		int leaf_count = BVHBuilders::build_sbvh<Triangle>(nodes[0], primitives, indices, nodes, node_index, 0, primitive_count, sah, temp, 1.0f / root_aabb.surface_area(), root_aabb);

		printf("SBVH Leaf count: %i\n", leaf_count);

		assert(node_index <= 2 * primitive_count);

		node_count = node_index;

		delete [] temp[0];
		delete [] temp[1];
		delete [] sah;
		
		primitive_count = leaf_count;
		PrimitiveType * new_primitives = new PrimitiveType[primitive_count];

		for (int i = 0; i < primitive_count; i++) {
			new_primitives[i] = primitives[indices_x[i]];
		}

		delete [] primitives;
		primitives = new_primitives;
	}

	inline void save_to_disk(const char * bvh_filename) const {
		FILE * file;
		fopen_s(&file, bvh_filename, "wb");

		fwrite(reinterpret_cast<const char *>(&primitive_count), sizeof(int), 1, file);
		fwrite(reinterpret_cast<const char *>(primitives), sizeof(PrimitiveType), primitive_count, file);

		fwrite(reinterpret_cast<const char *>(indices_x), sizeof(int), primitive_count, file);

		fwrite(reinterpret_cast<const char *>(&node_count), sizeof(int), 1, file);
		fwrite(reinterpret_cast<const char *>(nodes), sizeof(BVHNode), node_count, file);

		fclose(file);
	}

	inline void load_from_disk(const char * bvh_filename) {
		FILE * file;
		fopen_s(&file, bvh_filename, "rb"); 

		fread(reinterpret_cast<char *>(&primitive_count), sizeof(int), 1, file);

		primitives = new PrimitiveType[primitive_count];
		fread(reinterpret_cast<char *>(primitives), sizeof(PrimitiveType), primitive_count, file);
			
		indices_x = new int[primitive_count];
		fread(reinterpret_cast<char *>(indices_x), sizeof(int), primitive_count, file);

		fread(reinterpret_cast<char *>(&node_count), sizeof(int), 1, file);

		nodes = new BVHNode[node_count];
		fread(reinterpret_cast<char *>(nodes), sizeof(BVHNode), node_count, file);

		fclose(file);
	}
};
