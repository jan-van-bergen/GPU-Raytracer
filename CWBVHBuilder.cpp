#include "CWBVHBuilder.h"

#include <cstdlib>

int CWBVHBuilder::calculate_cost(int node_index, const BVHNode nodes[]) {
	const BVHNode & node = nodes[node_index];

	int num_primitives;

	if (node.is_leaf()) {
		num_primitives = node.get_count();
		assert(num_primitives == 1);

		// SAH cost
		float cost_leaf = node.aabb.surface_area() * float(num_primitives);

		for (int i = 0; i < 7; i++) {
			cost     [node_index * 7 + i]      = cost_leaf;
			decisions[node_index * 7 + i].type = Decision::Type::LEAF;
		}
	} else {
		num_primitives = 
			calculate_cost(node.left,     nodes) +
			calculate_cost(node.left + 1, nodes);

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

				decisions[node_index * 7].type = Decision::Type::LEAF;
			} else {
				cost[node_index * 7] = cost_internal;

				decisions[node_index * 7].type = Decision::Type::INTERNAL;
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
				decisions[node_index * 7 + i].type = Decision::Type::DISTRIBUTE;
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

void CWBVHBuilder::get_children(int node_index, const BVHNode nodes[], int i, int & child_count, int children[8]) {
	const BVHNode & node = nodes[node_index];
	
	if (node.is_leaf()) {
		children[child_count++] = node_index;

		return;
	}

	int distribute_0 = decisions[node_index * 7 + i].distribute_0;
	int distribute_1 = decisions[node_index * 7 + i].distribute_1;

	assert(distribute_0 >= 0 && distribute_0 < 7);
	assert(distribute_1 >= 0 && distribute_1 < 7);
	
	assert(child_count < 8);

	// Recurse on left child if it needs to distribute
	if (decisions[node.left * 7 + distribute_0].type == Decision::Type::DISTRIBUTE) {
		get_children(node.left, nodes, distribute_0, child_count, children);
	} else {
		children[child_count++] = node.left;
	}
	
	// Recurse on right child if it needs to distribute
	if (decisions[(node.left + 1) * 7 + distribute_1].type == Decision::Type::DISTRIBUTE) {
		get_children(node.left + 1, nodes, distribute_1, child_count, children);
	} else {
		children[child_count++] = node.left + 1;
	}
}

// Recursively count triangles in subtree of the given Node
// Simultaneously fills the indices buffer of the CWBVH
int CWBVHBuilder::count_primitives(int node_index, const BVHNode nodes[], const int indices[]) {
	const BVHNode & node = nodes[node_index];

	if (node.is_leaf()) {
		int primitive_count = node.get_count();
		assert(primitive_count == 1);

		for (int i = 0; i < primitive_count; i++) {
			cwbvh->indices[cwbvh->index_count++] = indices[node.first + i];
		}

		return primitive_count;
	}

	return 
		count_primitives(node.left,     nodes, indices) +
		count_primitives(node.left + 1, nodes, indices);
}

void CWBVHBuilder::order_children(int node_index, const BVHNode nodes[], int children[8], int child_count) {
	Vector3 p = nodes[node_index].aabb.get_center();

	float cost[8][8];

	// Fill cost table
	for (int c = 0; c < child_count; c++) {
		for (int s = 0; s < 8; s++) {
			Vector3 direction(
				(s & 0b100) ? -1.0f : 1.0f,
				(s & 0b010) ? -1.0f : 1.0f,
				(s & 0b001) ? -1.0f : 1.0f
			);

			cost[c][s] = Vector3::dot(nodes[children[c]].aabb.get_center() - p, direction);
		}
	}

	int   assignment[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
	bool slot_filled[8] = { };

	// Greedy child ordering, the paper mentions this as an alternative
	// that works about as well as the auction algorithm in practice
	while (true) {
		float min_cost = INFINITY;

		int min_slot  = -1;
		int min_index = -1;

		// Find cheapest unfilled slot of any unassigned child
		for (int c = 0; c < child_count; c++) {
			if (assignment[c] == -1) {
				for (int s = 0; s < 8; s++) {
					if (!slot_filled[s] && cost[s][c] < min_cost) {
						min_cost = cost[s][c];

						min_slot  = s;
						min_index = c;
					}
				}
			}
		}

		if (min_slot == -1) break;

		slot_filled[min_slot]  = true;
		assignment [min_index] = min_slot;
	}

	// Permute children array according to assignment
	int children_copy[8];
	memcpy(children_copy, children, sizeof(children_copy));

	for (int i = 0; i < 8; i++) {
		children[i] = -1;
	}

	for (int i = 0; i < child_count; i++) {
		assert(assignment   [i] != -1);
		assert(children_copy[i] != -1);

		children[assignment[i]] = children_copy[i];
	}
}

void CWBVHBuilder::collapse(const BVHNode nodes_sbvh[], const int indices_sbvh[], int node_index_cwbvh, int node_index_sbvh) {
	CWBVHNode  & node = cwbvh->nodes[node_index_cwbvh];
	const AABB & aabb = nodes_sbvh[node_index_sbvh].aabb;

	node.p = aabb.min;

	const int Nq = 8;
	const float denom = 1.0f / float((1 << Nq) - 1);

	Vector3 e(
		exp2f(ceilf(log2f((aabb.max.x - aabb.min.x) * denom))),
		exp2f(ceilf(log2f((aabb.max.y - aabb.min.y) * denom))),
		exp2f(ceilf(log2f((aabb.max.z - aabb.min.z) * denom)))
	);
	
	Vector3 one_over_e(1.0f / e.x, 1.0f / e.y, 1.0f / e.z);
	
	// Treat float as unsigned
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
	int children[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
	get_children(node_index_sbvh, nodes_sbvh, 0, child_count, children);

	assert(child_count <= 8);

	//order_children(node_index_sbvh, nodes_sbvh, children, child_count);

	node.imask = 0;

	node.base_index_child    = cwbvh->node_count;
	node.base_index_triangle = cwbvh->index_count;

	int node_internal_count = 0;
	int node_triangle_count = 0;

	for (int i = 0; i < 8; i++) {
		int child_index = children[i];

		if (child_index == -1) continue; // Empty slot

		const AABB & child_aabb = nodes_sbvh[child_index].aabb;

		node.quantized_min_x[i] = byte(floorf((child_aabb.min.x - node.p.x) * one_over_e.x));
		node.quantized_min_y[i] = byte(floorf((child_aabb.min.y - node.p.y) * one_over_e.y));
		node.quantized_min_z[i] = byte(floorf((child_aabb.min.z - node.p.z) * one_over_e.z));

		node.quantized_max_x[i] = byte(ceilf((child_aabb.max.x - node.p.x) * one_over_e.x));
		node.quantized_max_y[i] = byte(ceilf((child_aabb.max.y - node.p.y) * one_over_e.y));
		node.quantized_max_z[i] = byte(ceilf((child_aabb.max.z - node.p.z) * one_over_e.z));

		switch (decisions[child_index * 7].type) {
			case Decision::Type::LEAF: {
				int triangle_count = count_primitives(child_index, nodes_sbvh, indices_sbvh);
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

			case Decision::Type::INTERNAL: {
				node.meta[i] = (node_internal_count + 24) | 0b00100000;

				node.imask |= (1 << node_internal_count);

				cwbvh->node_count++;
				node_internal_count++;

				break;
			}

			default: abort();
		}
	}

	assert(node.base_index_child    + node_internal_count == cwbvh->node_count);
	assert(node.base_index_triangle + node_triangle_count == cwbvh->index_count);

	// Recurse on Internal Nodes
	for (int i = 0; i < 8; i++) {
		int child_index = children[i];
		if (child_index == -1) continue;

		if (decisions[child_index * 7].type == Decision::Type::INTERNAL) {
			collapse(nodes_sbvh, indices_sbvh, node.base_index_child + (node.meta[i] & 31) - 24, child_index);
		}
	}
}

void CWBVHBuilder::build(const BVH & bvh) {
	//ScopeTimer timer("Compressed Wide BVH Collapse");

	cwbvh->index_count = 0;
	cwbvh->indices     = new int[bvh.index_count];

	cwbvh->node_count = 1;
	cwbvh->nodes      = new CWBVHNode[bvh.node_count];

	// Fill cost table using dynamic programming (bottom up)
	calculate_cost(0, bvh.nodes);

	// Collapse SBVH into 8-way tree (top down)
	collapse(bvh.nodes, bvh.indices, 0, 0);

	assert(cwbvh->index_count == bvh.index_count);

	//printf("CWBVH Node Collapse: %i -> %i\n", bvh.node_count, cwbvh.node_count);
}
