#include "BVHOptimizer.h"

#include <queue>
#include <random>

#include "Util/Array.h"
#include "Util/BitArray.h"

#include "Util/ScopeTimer.h"

static std::random_device device;
static std::mt19937 rng(device());

// Calculates the SAH cost of a whole tree
static float bvh_sah_cost(const BVH & bvh) {
	float sum_leaf = 0.0f;
	float sum_node = 0.0f;

	for (int i = 0; i < bvh.node_count; i++) {
		if (i == 1) continue;

		const BVHNode & node = bvh.nodes[i];

		if (node.is_leaf()) {
			sum_leaf += node.aabb.surface_area() * node.get_count();
		} else {
			sum_node += node.aabb.surface_area();
		}
	}

	return (SAH_COST_NODE * sum_node + SAH_COST_LEAF * sum_leaf) / bvh.nodes[0].aabb.surface_area();
}

// Initialize array of parent indices by traversing the tree recursively
static void init_parent_indices(const BVH & bvh, int parent_indices[], int node_index = 0) {
	const BVHNode & node = bvh.nodes[node_index];

	if (node.is_leaf()) return;

	assert((node.left & 1) == 0);

	parent_indices[node.left    ] = node_index;
	parent_indices[node.left + 1] = node_index;

	init_parent_indices(bvh, parent_indices, node.left);
	init_parent_indices(bvh, parent_indices, node.left + 1);
}

// Produces a single batch consisting of 'batch_size' candidates for reinsertion based on random sampling
static void select_nodes_random(const BVH & bvh, const int parent_indices[], int batch_size, int batch_indices[]) {
	int offset = 0;
	int * temp = new int[bvh.node_count];

	// Identify Nodes that are valid for selection
	for (int i = 2; i < bvh.node_count; i++) {
		if (!bvh.nodes[i].is_leaf() && parent_indices[i] != 0) { // Node must be an internal Node and must have a grandparent (i.e. cannot be the child of the root)
			temp[offset++] = i;
		}
	}

	// Select a single batch of random Nodes from all viable Nodes
	std::sample(temp, temp + offset, batch_indices, batch_size, rng);

	delete [] temp;
}

// Produces a single batch consisting of 'batch_size' candidates for reinsertion based on which Nodes have the highest inefficiency measure
static void select_nodes_measure(const BVH & bvh, const int parent_indices[], int batch_size, int batch_indices[]) {
	float * costs = new float[bvh.node_count];

	int offset = 0;

	for (int i = 2; i < bvh.node_count; i++) {
		const BVHNode & node = bvh.nodes[i];

		if (!node.is_leaf() && parent_indices[i] != 0) { // Node must be an internal Node and must have a grandparent (i.e. cannot be the child of the root)
			float area = node.aabb.surface_area();
			float area_left  = bvh.nodes[node.left    ].aabb.surface_area();
			float area_right = bvh.nodes[node.left + 1].aabb.surface_area();

			float cost_sum  = 2.0f * area / (area_left + area_right);
			float cost_min  = area / Math::min(area_left, area_right);
			float cost_area = area;

			costs[i] = cost_sum * cost_min * cost_area;

			batch_indices[offset++] = i;
		}
	}

	// Select 'batch_size' worst Nodes
	std::partial_sort(batch_indices, batch_indices + batch_size, batch_indices + offset, [&](int a, int b) {
		return costs[a] > costs[b];
	});

	delete [] costs;
}

// Finds the global minimum of where best to insert the reinsertion node by traversing the tree using Branch and Bound
static void find_reinsertion(const BVH & bvh, const BVHNode & node_reinsert, float & min_cost, int & min_index) {
	float node_reinsert_area = node_reinsert.aabb.surface_area();

	// Compare based on induced cost
	auto cmp = [](const std::pair<int, float> & a, const std::pair<int, float> & b) {
		return a.second < b.second;
	};

	std::priority_queue<std::pair<int, float>, Array<std::pair<int, float>>, decltype(cmp)> priority_queue(cmp);

	priority_queue.emplace(0, 0.0f); // Push BVH root with 0 induced cost

	while (priority_queue.size() > 0) {
		auto [node_index, induced_cost] = priority_queue.top();
		priority_queue.pop();

		const BVHNode & node = bvh.nodes[node_index];

		if (induced_cost + node_reinsert_area >= min_cost) break; // Not possible to reduce min_cost, terminate

		float direct_cost = AABB::unify(node.aabb, node_reinsert.aabb).surface_area();
		float cost = induced_cost + direct_cost;

		if (cost < min_cost) {
			min_cost  = cost;
			min_index = node_index;
		}

		if (!node.is_leaf()) {
			float child_induced_cost = cost - node.aabb.surface_area();

			if (child_induced_cost + node_reinsert_area < min_cost) {
				priority_queue.emplace(node.left,     child_induced_cost);
				priority_queue.emplace(node.left + 1, child_induced_cost);
			}
		}
	}
}

// Update AABBs bottom up, until the root of the tree is reached
static void update_aabbs_bottom_up(BVH & bvh, int parent_indices[], int node_index) {
	assert(node_index >= 0);

	do {
		BVHNode & node = bvh.nodes[node_index];

		if (!node.is_leaf()) {
			node.aabb = AABB::unify(
				bvh.nodes[node.left    ].aabb,
				bvh.nodes[node.left + 1].aabb
			);
		}

		node_index = parent_indices[node_index];
	} while (node_index != -1);
}

// Calculates a split axis for the given BVH Node
// The axis on which the Node was originally split on becomes invalidated after reinsertion,
// which causes performance problems during traversal.
// This method selects a new split axis and swaps the child nodes such that the left child is also the leftmost node on that axis.
static void bvh_node_calc_axis(const BVH & bvh, int parent_indices[], int displacement[], int originated[], BVHNode & node) {
	int   max_axis = -1;
	float max_dist = 0.0f;

	Vector3 center_left  = bvh.nodes[node.left    ].aabb.get_center();
	Vector3 center_right = bvh.nodes[node.left + 1].aabb.get_center();

	// Calculate split axis based on a distance heuristic
	for (int dim = 0; dim < 3; dim++) {
		float dist =
			fabsf(bvh.nodes[node.left].aabb.min[dim] - bvh.nodes[node.left + 1].aabb.min[dim]) +
			fabsf(bvh.nodes[node.left].aabb.max[dim] - bvh.nodes[node.left + 1].aabb.max[dim]);

		if (dist >= max_dist) {
			max_dist = dist;
			max_axis = dim;
		}
	}

	assert(max_axis != -1);

	// Swap left and right children if needed
	if (center_left[max_axis] > center_right[max_axis]) {
		Util::swap(bvh.nodes[node.left], bvh.nodes[node.left + 1]);

		displacement[originated[node.left]]     = node.left + 1;
		displacement[originated[node.left + 1]] = node.left;

		// Update parent indices of grandchildren (if they exist) to account for the swap
		const BVHNode & child_left  = bvh.nodes[node.left];
		const BVHNode & child_right = bvh.nodes[node.left + 1];

		if (!child_left.is_leaf()) {
			parent_indices[child_left.left]     = node.left;
			parent_indices[child_left.left + 1] = node.left;
		}

		if (!child_right.is_leaf()) {
			parent_indices[child_right.left]     = node.left + 1;
			parent_indices[child_right.left + 1] = node.left + 1;
		}
	}

	static const int dim_bits[3] = {
		BVH_AXIS_X_BITS,
		BVH_AXIS_Y_BITS,
		BVH_AXIS_Z_BITS
	};

	node.count = dim_bits[max_axis];
}

struct CollapseCost {
	int primitive_count;
	float sah;
};

// Bottom up calculation of the cost of collapsing multiple leaf nodes into one
static CollapseCost bvh_calc_collapse_cost(const BVH & bvh, BitArray & collapse, int node_index = 0) {
	const BVHNode & node = bvh.nodes[node_index];

	if (node.is_leaf()) {
		int count = node.get_count();

		return { count, float(count) * SAH_COST_LEAF };
	} else {
		CollapseCost cost_left  = bvh_calc_collapse_cost(bvh, collapse, node.left);
		CollapseCost cost_right = bvh_calc_collapse_cost(bvh, collapse, node.left + 1);

		int total_primtive_count = cost_left.primitive_count + cost_right.primitive_count;

		float sah_leaf = SAH_COST_LEAF * float(total_primtive_count);
		float sah_node = SAH_COST_NODE + (
			bvh.nodes[node.left    ].aabb.surface_area() * cost_left .sah +
			bvh.nodes[node.left + 1].aabb.surface_area() * cost_right.sah
		) / node.aabb.surface_area();

		if (sah_leaf < sah_node) {
			assert(!collapse[node_index]);
			collapse[node_index] = true;

			return { total_primtive_count, sah_leaf };
		} else {
			return { total_primtive_count, sah_node };
		}
	}
}

// Helper method that collapses all subnodes in a given subtree into a single leaf Node
static int collapse_subtree(const BVH & bvh, BVH & new_bvh, int node_index) {
	const BVHNode & node = bvh.nodes[node_index];

	if (node.is_leaf()) {
		int count = node.get_count();

		for (int i = 0; i < count; i++) {
			new_bvh.indices[new_bvh.index_count++] = bvh.indices[node.first + i];
		}

		return count;
	} else {
		int count_left  = collapse_subtree(bvh, new_bvh, node.left);
		int count_right = collapse_subtree(bvh, new_bvh, node.left + 1);

		return count_left + count_right;
	}
};

// Collapse leaf nodes based on precalculated cost
static void bvh_collapse(const BVH & bvh, BVH & new_bvh, int new_index, BitArray & collapse, int node_index = 0) {
	const BVHNode & node = bvh.nodes[node_index];

	BVHNode & new_node = new_bvh.nodes[new_index];
	new_node.aabb  = node.aabb;
	new_node.count = node.count;

	if (node.is_leaf()) {
		int count = node.get_count();

		new_node.first = new_bvh.index_count;

		for (int i = 0; i < count; i++) {
			new_bvh.indices[new_bvh.index_count++] = bvh.indices[node.first + i];
		}

		assert(new_node.is_leaf());
	} else {
		// Check if this internal Node needs to collapse its subtree into a leaf
		if (collapse[node_index]) {
			new_node.count = collapse_subtree(bvh, new_bvh, node_index);
			new_node.first = new_bvh.index_count - new_node.count;

			assert(new_node.is_leaf());
		} else {
			new_node.left = new_bvh.node_count;
			new_bvh.node_count += 2;

			assert(!new_node.is_leaf());

			bvh_collapse(bvh, new_bvh, new_node.left,     collapse, node.left);
			bvh_collapse(bvh, new_bvh, new_node.left + 1, collapse, node.left + 1);
		}
	}
}

void BVHOptimizer::optimize(BVH & bvh) {
	ScopeTimer timer("BVH Optimization");

	float cost_before = bvh_sah_cost(bvh);

	if (bvh.node_count < 8) return; // Tree too small to optimize

	int * parent_indices = new int[bvh.node_count];
	parent_indices[0] = -1; // Root has no parent
	init_parent_indices(bvh, parent_indices);

	constexpr int P_R = 5;  // After P_R batches with no improvement to the best SAH cost we switch to random Node selection
	constexpr int P_T = 10; // After P_T batches with no improvement to the best SAH cost we terminate the algorithm
	constexpr int k = 100;

	static_assert(P_T >= P_R);

	int   batch_size = bvh.node_count / k;
	int * batch_indices = new int[bvh.node_count];

	int batch_count = 0;
	int batches_since_last_cost_reduction = 0;

	float sah_cost_best = cost_before;

	enum struct NodeSelectionMethod {
		RANDOM,
		MEASURE
	} node_selection_method = NodeSelectionMethod::MEASURE;

	int * originated   = new int[bvh.node_count];
	int * displacement = new int[bvh.node_count];

	clock_t start_time = clock();

	while (true) {
		// Select a batch of internal Nodes, either randomly or using a heuristic measure
		switch (node_selection_method) {
			case NodeSelectionMethod::RANDOM:  select_nodes_random (bvh, parent_indices, batch_size, batch_indices); break;
			case NodeSelectionMethod::MEASURE: select_nodes_measure(bvh, parent_indices, batch_size, batch_indices); break;

			default: abort();
		}

		for (int i = 0; i < bvh.node_count; i++) {
			originated  [i] = i;
			displacement[i] = i;
		}

		for (int i = 0; i < batch_size; i++) {
			int node_index = displacement[batch_indices[i]];
			if (node_index == -1) continue; // This Node was overwritten by another reinsertion and no longer exists

			const BVHNode & node = bvh.nodes[node_index];

			int parent        = parent_indices[node_index];
			int parent_parent = parent_indices[parent];

			if (node.is_leaf() || parent == 0 || parent == -1) continue;

			int sibling = (node_index & 1) ? node_index - 1 : node_index + 1; // Other child of the same parent as current node

			int child_left  = node.left;
			int child_right = node.left + 1;

			struct Reinsert {
				int     node_index;
				BVHNode node;
			};

			int      nodes_unused  [2];
			Reinsert nodes_reinsert[2];

			// Child with largest area should be reinserted first
			float area_left  = bvh.nodes[node.left]    .aabb.surface_area();
			float area_right = bvh.nodes[node.left + 1].aabb.surface_area();

			if (area_left > area_right) {
				nodes_reinsert[0] = { child_left,  bvh.nodes[child_left]  };
				nodes_reinsert[1] = { child_right, bvh.nodes[child_right] };
			} else {
				nodes_reinsert[0] = { child_right, bvh.nodes[child_right] };
				nodes_reinsert[1] = { child_left,  bvh.nodes[child_left]  };
			}

			nodes_unused[0] = node_index & ~1;
			nodes_unused[1] = child_left;

			// Keep tree topologically consistent
			bvh.nodes[parent] = bvh.nodes[sibling];
			parent_indices[sibling] = parent_parent;

			displacement[originated[sibling]] = parent;
			originated[parent] = originated[sibling];

			if (!bvh.nodes[sibling].is_leaf()) {
				parent_indices[bvh.nodes[sibling].left    ] = parent;
				parent_indices[bvh.nodes[sibling].left + 1] = parent;
			}

			displacement[originated[parent]]     = -1;
			displacement[originated[node_index]] = -1;

			update_aabbs_bottom_up(bvh, parent_indices, parent_parent);

			// Reinsert Nodes
			for (int j = 0; j < 2; j++) {
				int      unused   = nodes_unused  [j];
				Reinsert reinsert = nodes_reinsert[j];

				assert((unused & 1) == 0);

				// Find the best position to reinsert the given Node
				float min_cost  = INFINITY;
				int   min_index = -1;

				find_reinsertion(bvh, reinsert.node, min_cost, min_index);

				// Bookkeeping updates to perform the reinsertion
				bvh.nodes[unused    ] = bvh.nodes[min_index];
				bvh.nodes[unused + 1] = reinsert.node;

				parent_indices[unused    ] = min_index;
				parent_indices[unused + 1] = min_index;

				displacement[originated[min_index]]           = unused;
				displacement[originated[reinsert.node_index]] = unused + 1;

				originated[unused    ] = originated[min_index];
				originated[unused + 1] = originated[reinsert.node_index];

				if (!bvh.nodes[min_index].is_leaf()) {
					parent_indices[bvh.nodes[min_index].left    ] = unused;
					parent_indices[bvh.nodes[min_index].left + 1] = unused;
				}

				if (!reinsert.node.is_leaf()) {
					parent_indices[reinsert.node.left    ] = unused + 1;
					parent_indices[reinsert.node.left + 1] = unused + 1;
				}

				bvh.nodes[min_index].left  = unused;
				bvh.nodes[min_index].count = 0;

				update_aabbs_bottom_up(bvh, parent_indices, min_index);

				bvh_node_calc_axis(bvh, parent_indices, displacement, originated, bvh.nodes[min_index]);

				assert(!bvh.nodes[min_index].is_leaf());
			}
		}

		float sah_cost = bvh_sah_cost(bvh);

		if (sah_cost < sah_cost_best) {
			sah_cost_best = sah_cost;

			batches_since_last_cost_reduction = 0;

			node_selection_method = NodeSelectionMethod::MEASURE;
		} else {
			batches_since_last_cost_reduction++;

			// Check if we should switch to random selection
			if (batches_since_last_cost_reduction == P_R) {
				node_selection_method = NodeSelectionMethod::RANDOM;
			}
			// Check if we should terminate
			if (batches_since_last_cost_reduction == P_T) {
				break;
			}
		}

		clock_t curr_time = clock();
		size_t  duration  = (curr_time - start_time) * 1000 / CLOCKS_PER_SEC;

		if (duration >= BVH_OPTIMIZER_MAX_TIME || batch_count >= BVH_OPTIMIZER_MAX_NUM_BATCHES) {
			break;
		}

		printf("%i: SAH=%f best=%f last_reduction=%i     \r", batch_count, sah_cost, sah_cost_best, batches_since_last_cost_reduction);
		batch_count++;
	}

	delete [] originated;
	delete [] displacement;

	delete [] batch_indices;
	delete [] parent_indices;

	// Collapse leaf Nodes of the tree based on SAH cost
	// Step 1: Calculate costs of collapse, and fill array with the decision to collapse, yes or no
	BitArray collapse;
	collapse.init(bvh.node_count);
	collapse.set_all(false);

#if BVH_TYPE != BVH_CWBVH // CWBVH also requires every leaf Node to contain only 1 primitive, so in that case we skip the collapse step
	bvh_calc_collapse_cost(bvh, collapse);
#endif

	// Collapse BVH using a copy
	BVH new_bvh;
	new_bvh.nodes   = new BVHNode[bvh.node_count];
	new_bvh.indices = new int    [bvh.index_count];
	new_bvh.node_count  = 2;
	new_bvh.index_count = 0;

	bvh_collapse(bvh, new_bvh, 0, collapse);

	assert(new_bvh.node_count  <= bvh.node_count);
	assert(new_bvh.index_count == bvh.index_count);

	// Cleanup
	collapse.free();

	delete [] bvh.nodes;
	delete [] bvh.indices;

	bvh = new_bvh;

	// Report the improvement of the SAH cost
	float cost_after = bvh_sah_cost(bvh);
	printf("\ncost: %f -> %f\n", cost_before, cost_after);
}
