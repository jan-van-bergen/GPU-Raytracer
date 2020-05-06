#pragma once
#include "BVH.h"

#include "ScopedTimer.h"

typedef unsigned char byte;

struct CompressedWideBVHNode {
	Vector3 p;
	byte e[3];
	byte imask;

	unsigned base_index_child;
	unsigned base_index_triangle;

	byte meta           [8];
	byte quantized_min_x[8];
	byte quantized_min_y[8];
	byte quantized_min_z[8];
	byte quantized_max_x[8];
	byte quantized_max_y[8];
	byte quantized_max_z[8];
};

static_assert(sizeof(CompressedWideBVHNode) == 80);

struct CWBVHDecision {
	enum class Type : char {
		LEAF,
		INTERNAL,
		DISTRIBUTE
	} type;

	char distribute_0 = -1;
	char distribute_1 = -1;

	// char padding;
};

inline int calculate_cost(float cost[], CWBVHDecision decisions[], int node_index, const BVHNode nodes[]) {
	const BVHNode & node = nodes[node_index];

	int num_primitives;

	if (node.is_leaf()) {
		num_primitives = node.get_count();
		assert(num_primitives == 1);

		// SAH cost
		float cost_leaf = node.aabb.surface_area() * float(num_primitives);

		for (int i = 0; i < 7; i++) {
			cost     [node_index * 7 + i]      = cost_leaf;
			decisions[node_index * 7 + i].type = CWBVHDecision::Type::LEAF;
		}
	} else {
		num_primitives = 
			calculate_cost(cost, decisions, node.left,     nodes) +
			calculate_cost(cost, decisions, node.left + 1, nodes);

		// Separate case: i=0 (i=1 in the paper)
		{
			float cost_leaf = num_primitives <= 3 ? float(num_primitives) * node.aabb.surface_area() : INFINITY;

			float cost_distribute = INFINITY;
			
			char distribute_0 = -1;
			char distribute_1 = -1;

			for (int k = 0; k < 7; k++) {
				float c = 
					cost[(node.left)     * 7 +     k] + 
					cost[(node.left + 1) * 7 + 6 - k];

				if (c < cost_distribute) {
					cost_distribute = c;
					
					distribute_0 =     k;
					distribute_1 = 6 - k;
				}
			}

			float cost_internal = cost_distribute + node.aabb.surface_area();

			if (cost_leaf < cost_internal) {
				cost[node_index * 7] = cost_leaf;

				decisions[node_index * 7].type = CWBVHDecision::Type::LEAF;
			} else {
				cost[node_index * 7] = cost_internal;

				decisions[node_index * 7].type = CWBVHDecision::Type::INTERNAL;
			}

			decisions[node_index * 7].distribute_0 = distribute_0;
			decisions[node_index * 7].distribute_1 = distribute_1;
		}

		// In the paper i=2..7
		for (int i = 1; i < 7; i++) {
			float cost_distribute = cost[node_index * 7 + i - 1];

			char distribute_0 = -1;
			char distribute_1 = -1;

			for (int k = 0; k < i; k++) {
				float c = 
					cost[(node.left)     * 7 +     k    ] + 
					cost[(node.left + 1) * 7 + i - k - 1];

				if (c < cost_distribute) {
					cost_distribute = c;
					
					distribute_0 =     k;
					distribute_1 = i - k - 1;
				}
			}

			cost[node_index * 7 + i] = cost_distribute;

			if (distribute_0 != -1) {
				decisions[node_index * 7 + i].type = CWBVHDecision::Type::DISTRIBUTE;
				decisions[node_index * 7 + i].distribute_0 = distribute_0;
				decisions[node_index * 7 + i].distribute_1 = distribute_1;
			} else {
				decisions[node_index * 7 + i].type         = decisions[node_index * 7 + i - 1].type;
				decisions[node_index * 7 + i].distribute_0 = decisions[node_index * 7 + i - 1].distribute_0;
				decisions[node_index * 7 + i].distribute_1 = decisions[node_index * 7 + i - 1].distribute_1;
			}
		}
	}

	return num_primitives;
}

inline void get_children(const BVHNode nodes[], const CWBVHDecision decisions[], int node_index, int i, int & child_count, int children[8]) {
	const BVHNode & node = nodes[node_index];

	int child_indices[2] = { node.left, node.left + 1 };
	int distributes[2] = {
		decisions[node_index * 7 + i].distribute_0,
		decisions[node_index * 7 + i].distribute_1
	};

	assert(distributes[0] >= 0 && distributes[0] < 7);
	assert(distributes[1] >= 0 && distributes[1] < 7);

	for (int c = 0; c < 2; c++) {
		if (decisions[child_indices[c] * 7 + distributes[c]].type == CWBVHDecision::Type::DISTRIBUTE) {
			get_children(nodes, decisions, child_indices[c], distributes[c], child_count, children);
		} else {
			assert(child_count < 8);

			children[child_count++] = child_indices[c];
		}
	}
}

inline int count_primitives(const BVHNode nodes[], int node_index, int & index_count, int indices_wbvh[], const int indices_sbvh[]) {
	const BVHNode & node = nodes[node_index];

	if (node.is_leaf()) {
		int primitive_count = node.get_count();
		assert(primitive_count == 1);

		for (int i = 0; i < primitive_count; i++) {
			indices_wbvh[index_count++] = indices_sbvh[node.first + i];
		}

		return primitive_count;
	}

	return 
		count_primitives(nodes, node.left,     index_count, indices_wbvh, indices_sbvh) +
		count_primitives(nodes, node.left + 1, index_count, indices_wbvh, indices_sbvh);
}

inline void collapse(int & node_count, CompressedWideBVHNode nodes_wbvh[], int & index_count, int indices_wbvh[], const BVHNode nodes_sbvh[], const int indices_sbvh[], const CWBVHDecision decisions[], int node_index_wbvh, int node_index_sbvh) {
	CompressedWideBVHNode & node = nodes_wbvh[node_index_wbvh];

	const AABB & aabb = nodes_sbvh[node_index_sbvh].aabb;

	node.p = aabb.min;

	const int Nq = 8;
	const float denom = 1.0f / float((1 << Nq) - 1);

	Vector3 e(
		exp2f(ceilf(log2f((aabb.max.x - aabb.min.x) * denom))),
		exp2f(ceilf(log2f((aabb.max.y - aabb.min.y) * denom))),
		exp2f(ceilf(log2f((aabb.max.z - aabb.min.z) * denom)))
	);

	unsigned u_ex, u_ey, u_ez;
	memcpy(&u_ex, &e.x, 4);
	memcpy(&u_ey, &e.y, 4);
	memcpy(&u_ez, &e.z, 4);

	// Only the exponent bits can be non-zero
	assert((u_ex & 0b10000000011111111111111111111111) == 0);
	assert((u_ey & 0b10000000011111111111111111111111) == 0);
	assert((u_ez & 0b10000000011111111111111111111111) == 0);

	// Store only 8 bit exponent
	node.e[0] = u_ex >> 23;
	node.e[1] = u_ey >> 23;
	node.e[2] = u_ez >> 23;
	
	int child_count = 0;
	int children[8];
	get_children(nodes_sbvh, decisions, node_index_sbvh, 0, child_count, children);

	assert(child_count <= 8);

	//////////////////////////
	// @TODO: rank children //
	//////////////////////////

	Vector3 one_over_e(1.0f / e.x, 1.0f / e.y, 1.0f / e.z);
	
	for (int i = 0; i < 8; i++) {
		node.meta[i] = 0;
	}

	node.imask = 0;

	node.base_index_child    = node_count;
	node.base_index_triangle = index_count;

	int node_internal_count = 0;
	int node_triangle_count = 0;

	for (int i = 0; i < child_count; i++) {
		int child_index = children[i];

		const AABB & child_aabb = nodes_sbvh[child_index].aabb;

		node.quantized_min_x[i] = byte(floorf((child_aabb.min.x - node.p.x) * one_over_e.x));
		node.quantized_min_y[i] = byte(floorf((child_aabb.min.y - node.p.y) * one_over_e.y));
		node.quantized_min_z[i] = byte(floorf((child_aabb.min.z - node.p.z) * one_over_e.z));

		node.quantized_max_x[i] = byte(ceilf((child_aabb.max.x - node.p.x) * one_over_e.x));
		node.quantized_max_y[i] = byte(ceilf((child_aabb.max.y - node.p.y) * one_over_e.y));
		node.quantized_max_z[i] = byte(ceilf((child_aabb.max.z - node.p.z) * one_over_e.z));

		switch (decisions[child_index * 7].type) {
			case CWBVHDecision::Type::LEAF: {
				int triangle_count = count_primitives(nodes_sbvh, child_index, index_count, indices_wbvh, indices_sbvh);
				assert(triangle_count > 0 && triangle_count <= 3);

				// Three highest bits contain unary representation of triangle count
				for (int j = 0; j < triangle_count; j++) {
					node.meta[i] |= (1 << (j + 5));
				}

				node.meta[i] |= node_triangle_count;

				node_triangle_count += triangle_count;
				assert(node_triangle_count <= 24);

				break;
			}

			case CWBVHDecision::Type::INTERNAL: {
				node.meta[i] = (node_internal_count + 24) | 0b00100000;

				node.imask |= (1 << node_internal_count);

				node_count++;
				node_internal_count++;

				break;
			}

			default: abort();
		}
	}

	assert(node.base_index_child    + node_internal_count == node_count);
	assert(node.base_index_triangle + node_triangle_count == index_count);

	// Recurse on Internal Nodes
	for (int i = 0; i < child_count; i++) {
		int child_index = children[i];

		if (decisions[child_index * 7].type == CWBVHDecision::Type::INTERNAL) {
			collapse(node_count, nodes_wbvh, index_count, indices_wbvh, nodes_sbvh, indices_sbvh, decisions, node.base_index_child + (node.meta[i] & 31) - 24, child_index);
		}
	}
}

struct CompressedWideBVH {
	int        triangle_count;
	Triangle * triangles;
	
	int * indices;

	int                     node_count;
	CompressedWideBVHNode * nodes;

	int leaf_count;
	
	inline void init(const BVH & bvh) {
		ScopedTimer timer("Compressed Wide BVH Construction");

		triangle_count = bvh.triangle_count;
		triangles      = bvh.triangles;
		
		leaf_count = 0;
		indices    = new int[bvh.leaf_count * 2];

		node_count = 1;
		nodes      = new CompressedWideBVHNode[bvh.node_count];

		float         * cost      = new float        [bvh.node_count * 7];
		CWBVHDecision * decisions = new CWBVHDecision[bvh.node_count * 7];

		calculate_cost(cost, decisions, 0, bvh.nodes);

		collapse(node_count, nodes, leaf_count, indices, bvh.nodes, bvh.indices, decisions, 0, 0);

		printf("CWBVH Node Collapse: %i -> %i\n\n", bvh.node_count, node_count);

		delete [] decisions;
		delete [] cost;
	}
};