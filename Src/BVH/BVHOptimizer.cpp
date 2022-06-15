#include "BVHOptimizer.h"

#include "Config.h"

#include "Core/IO.h"
#include "Core/Array.h"
#include "Core/MinHeap.h"
#include "Core/Timer.h"
#include "Core/Random.h"
#include "Core/Allocators/LinearAllocator.h"

#include "Util/Util.h"

// Calculates the SAH cost of a whole tree
static float bvh_sah_cost(const BVH2 & bvh) {
	float sum_leaf = 0.0f;
	float sum_node = 0.0f;

	for (size_t i = 0; i < bvh.nodes.size(); i++) {
		if (i == 1) continue;

		const BVHNode2 & node = bvh.nodes[i];

		if (node.is_leaf()) {
			sum_leaf += node.aabb.surface_area() * node.count;
		} else {
			sum_node += node.aabb.surface_area();
		}
	}

	return (
		cpu_config.sah_cost_node * sum_node +
		cpu_config.sah_cost_leaf * sum_leaf
	) / bvh.nodes[0].aabb.surface_area();
}

// Initialize array of parent indices
static Array<int> get_parent_indices(const BVH2 & bvh, Allocator * allocator) {
	Array<int> parent_indices(bvh.nodes.size(), allocator);
	parent_indices[0] = INVALID; // Root has no parent

	for (size_t i = 2; i < bvh.nodes.size(); i++) {
		const BVHNode2 & node = bvh.nodes[i];
		if (node.is_leaf()) {
			ASSERT(node.count == 1);
		} else {
			ASSERT((node.left & 1) == 0);

			parent_indices[node.left]     = i;
			parent_indices[node.left + 1] = i;
		}
	}

	return parent_indices;
}

// Produces a single batch consisting of 'batch_size' candidates for reinsertion based on random sampling
static void select_nodes_random(const BVH2 & bvh, const Array<int> & parent_indices, int batch_size, Allocator * allocator, Array<int> & batch_indices, RNG & rng) {
	size_t offset = 0;
	Array<int> temp(bvh.nodes.size(), allocator);

	// Identify Nodes that are valid for selection
	for (size_t i = 2; i < bvh.nodes.size(); i++) {
		if (!bvh.nodes[i].is_leaf() && parent_indices[i] != 0) { // Node must be an internal Node and must have a grandparent (i.e. cannot be the child of the root)
			temp[offset++] = int(i);
		}
	}

	Random::sample(temp.data(), temp.data() + offset, batch_indices.data(), batch_indices.data() + batch_size, rng);
}

// Produces a single batch consisting of 'batch_size' candidates for reinsertion based on which Nodes have the highest inefficiency measure
static void select_nodes_measure(const BVH2 & bvh, const Array<int> & parent_indices, int batch_size, Allocator * allocator, Array<int> & batch_indices) {
	Array<float> costs(bvh.nodes.size(), allocator);

	int offset = 0;

	for (size_t i = 2; i < bvh.nodes.size(); i++) {
		const BVHNode2 & node = bvh.nodes[i];

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

	auto cmp = [costs](int a, int b) {
		return costs[a] > costs[b];
	};
	MinHeap<int, decltype(cmp)> heap(cmp, allocator);

	for (int i = 0; i < offset; i++) {
		heap.insert(batch_indices[i]);
	}
	for (int i = 0; i < batch_size; i++) {
		batch_indices[i] = heap.pop();
	}
}

// Finds the global minimum of where best to insert the reinsertion node by traversing the tree using Branch and Bound
static void find_reinsertion(const BVH2 & bvh, const BVHNode2 & node_reinsert, Allocator * allocator, float & min_cost, int & min_index) {
	float node_reinsert_area = node_reinsert.aabb.surface_area();

	struct Pair {
		int   node_index;
		float induced_cost;

		bool operator<(Pair other) const {
			return induced_cost < other.induced_cost; // Compare based on induced cost
		}
	};

	MinHeap<Pair> priority_queue(allocator);
	priority_queue.emplace(0, 0.0f); // Push BVH root with 0 induced cost

	while (priority_queue.size() > 0) {
		auto [node_index, induced_cost] = priority_queue.pop();

		const BVHNode2 & node = bvh.nodes[node_index];

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
static void update_aabbs_bottom_up(BVH2 & bvh, const Array<int> & parent_indices, int node_index) {
	ASSERT(node_index >= 0);

	do {
		BVHNode2 & node = bvh.nodes[node_index];

		if (!node.is_leaf()) {
			node.aabb = AABB::unify(
				bvh.nodes[node.left    ].aabb,
				bvh.nodes[node.left + 1].aabb
			);
		}

		node_index = parent_indices[node_index];
	} while (node_index != INVALID);
}

// Calculates a split axis for the given BVH Node
// The axis on which the Node was originally split on becomes invalidated after reinsertion,
// which causes performance problems during traversal.
// This method selects a new split axis and swaps the child nodes such that the left child is also the leftmost node on that axis.
static void bvh_node_calc_axis(BVH2 & bvh, Array<int> & parent_indices, Array<int> & displacement, const Array<int> & originated, BVHNode2 & node) {
	int   max_axis = INVALID;
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

	ASSERT(max_axis != INVALID);

	// Swap left and right children if needed
	if (center_left[max_axis] > center_right[max_axis]) {
		Util::swap(bvh.nodes[node.left], bvh.nodes[node.left + 1]);

		displacement[originated[node.left]]     = node.left + 1;
		displacement[originated[node.left + 1]] = node.left;

		// Update parent indices of grandchildren (if they exist) to account for the swap
		const BVHNode2 & child_left  = bvh.nodes[node.left];
		const BVHNode2 & child_right = bvh.nodes[node.left + 1];

		if (!child_left.is_leaf()) {
			parent_indices[child_left.left]     = node.left;
			parent_indices[child_left.left + 1] = node.left;
		}

		if (!child_right.is_leaf()) {
			parent_indices[child_right.left]     = node.left + 1;
			parent_indices[child_right.left + 1] = node.left + 1;
		}
	}

	node.count = 0;
	node.axis  = max_axis;
}

void BVHOptimizer::optimize(BVH2 & bvh) {
	// Calculate the number of BHV Nodes that may be included in a batch
	// These Nodes must be internal Nodes and must have a grandparent (i.e. cannot be the child of the root)
	// This means a tree with 7 nodes has 0 batch candidates, and every 2 additional child nodes in the tree account for 1 more batch candidate:
	int num_batch_candidates = Math::max<int>((bvh.nodes.size() - 7) / 2, 0);
	if (num_batch_candidates < 8) {
		return; // Too small to optimize
	}

	ScopeTimer timer("BVH Optimization"_sv);

	float cost_before = bvh_sah_cost(bvh);

	LinearAllocator<MEGABYTES(1)> init_allocator; // Memory used during the entire optimization process
	LinearAllocator<MEGABYTES(1)> loop_allocator; // Memory reset every batch iteration

	Array<int> parent_indices = get_parent_indices(bvh, &init_allocator);

	constexpr int P_R = 5;  // After P_R batches with no improvement to the best SAH cost we switch to random Node selection
	constexpr int P_T = 10; // After P_T batches with no improvement to the best SAH cost we terminate the algorithm
	constexpr int k = 100;

	static_assert(P_T >= P_R);

	int        batch_size = Math::max<int>(bvh.nodes.size() / k, num_batch_candidates);
	Array<int> batch_indices(bvh.nodes.size(), &init_allocator);

	int batch_count = 0;
	int batches_since_last_cost_reduction = 0;

	float sah_cost_best = cost_before;

	enum struct NodeSelectionMethod {
		RANDOM,
		MEASURE
	} node_selection_method = NodeSelectionMethod::MEASURE;

	Array<int> originated  (bvh.nodes.size(), &init_allocator);
	Array<int> displacement(bvh.nodes.size(), &init_allocator);

	RNG rng(time(nullptr));

	clock_t start_time = clock();

	while (true) {
		loop_allocator.reset();

		// Select a batch of internal Nodes, either randomly or using a heuristic measure
		switch (node_selection_method) {
			case NodeSelectionMethod::RANDOM:  select_nodes_random (bvh, parent_indices, batch_size, &loop_allocator, batch_indices, rng); break;
			case NodeSelectionMethod::MEASURE: select_nodes_measure(bvh, parent_indices, batch_size, &loop_allocator, batch_indices);      break;
			default: ASSERT_UNREACHABLE();
		}

		for (size_t i = 0; i < bvh.nodes.size(); i++) {
			originated  [i] = i;
			displacement[i] = i;
		}

		for (int i = 0; i < batch_size; i++) {
			int node_index = displacement[batch_indices[i]];
			if (node_index == INVALID) continue; // This Node was overwritten by another reinsertion and no longer exists

			const BVHNode2 & node = bvh.nodes[node_index];

			int parent        = parent_indices[node_index];
			int parent_parent = parent_indices[parent];

			if (node.is_leaf() || parent == 0 || parent == INVALID) continue;

			int sibling = (node_index & 1) ? node_index - 1 : node_index + 1; // Other child of the same parent as current node

			int child_left  = node.left;
			int child_right = node.left + 1;

			struct Reinsert {
				int      node_index;
				BVHNode2 node;
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

			displacement[originated[parent]]     = INVALID;
			displacement[originated[node_index]] = INVALID;

			update_aabbs_bottom_up(bvh, parent_indices, parent_parent);

			// Reinsert Nodes
			for (int j = 0; j < 2; j++) {
				int      unused   = nodes_unused  [j];
				Reinsert reinsert = nodes_reinsert[j];

				ASSERT((unused & 1) == 0);

				// Find the best position to reinsert the given Node
				float min_cost  = INFINITY;
				int   min_index = INVALID;

				find_reinsertion(bvh, reinsert.node, &loop_allocator, min_cost, min_index);

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

				ASSERT(!bvh.nodes[min_index].is_leaf());
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

		if (duration >= cpu_config.bvh_optimizer_max_time || batch_count >= cpu_config.bvh_optimizer_max_num_batches) {
			break;
		}

		IO::print("{}: SAH={} best={} last_reduction={}     \r"_sv, batch_count, sah_cost, sah_cost_best, batches_since_last_cost_reduction);
		batch_count++;
	}

	// Report the improvement of the SAH cost
	float cost_after = bvh_sah_cost(bvh);
	IO::print("\ncost: {} -> {}\n"_sv, cost_before, cost_after);
}
