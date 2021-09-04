#include "SBVHBuilder.h"

#include "Config.h"

#include "BVHPartitions.h"

#include "Util/Util.h"
#include "Util/ScopeTimer.h"

int SBVHBuilder::build_sbvh(BVHNode2 & node, const Triangle * triangles, PrimitiveRef * indices[3], int & node_index, int first_index, int index_count, float inv_root_surface_area) {
	if (index_count == 1) {
		// Leaf Node, terminate recursion
		node.first = first_index;
		node.count = index_count;

		return node.count;
	}

	// Object Split information
	BVHPartitions::ObjectSplit object_split = BVHPartitions::partition_object(indices, first_index, index_count, bounds, sah);
	assert(object_split.index != INVALID);

	// Calculate the overlap between the child bounding boxes resulting from the Object Split
	AABB overlap = AABB::overlap(object_split.aabb_left, object_split.aabb_right);
	float lamba = overlap.is_valid() ? overlap.surface_area() : 0.0f;

	// Divide by the surface area of the bounding box of the root Node
	float ratio = lamba * inv_root_surface_area;
	assert(ratio >= 0.0f && ratio <= 1.0f);

	BVHPartitions::SpatialSplit spatial_split;

	// If ratio between overlap area and root area is large enough, consider a Spatial Split
	if (ratio > config.sbvh_alpha) {
		spatial_split = BVHPartitions::partition_spatial(triangles, indices, first_index, index_count, sah, node.aabb);
	} else {
		spatial_split.cost = INFINITY;
	}

	assert(object_split.cost != INFINITY || spatial_split.cost != INFINITY);

	if (index_count <= max_primitives_in_leaf) {
		// Check SAH termination condition
		float leaf_cost = node.aabb.surface_area() * config.sah_cost_leaf * float(index_count);
		float node_cost = node.aabb.surface_area() * config.sah_cost_node + Math::min(object_split.cost, spatial_split.cost);

		if (leaf_cost < node_cost) {
			node.first = first_index;
			node.count = index_count;

			return node.count;
		}
	}

	node.left = node_index;
	node_index += 2;

	PrimitiveRef * children_left [3] {
		indices[0] + first_index,
		indices[1] + first_index,
		indices[2] + first_index
	};
	PrimitiveRef * children_right[3] {
		indices_going_right_x + indices_going_right_offset,
		indices_going_right_y + indices_going_right_offset,
		indices_going_right_z + indices_going_right_offset
	};

	int children_left_count [3] = { 0, 0, 0 };
	int children_right_count[3] = { 0, 0, 0 };

	int n_left, n_right;

	AABB child_aabb_left;
	AABB child_aabb_right;

	// The two temp arrays will be used as lookup tables
	if (object_split.cost <= spatial_split.cost) {
		// Perform Object Split

		node.count = 0;
		node.axis  = object_split.dimension;

		for (int i = first_index;        i < object_split.index;        i++) indices_going_left[indices[object_split.dimension][i].index] = true;
		for (int i = object_split.index; i < first_index + index_count; i++) indices_going_left[indices[object_split.dimension][i].index] = false;

		for (int dimension = 0; dimension < 3; dimension++) {
			for (int i = first_index; i < first_index + index_count; i++) {
				bool goes_left = indices_going_left[indices[dimension][i].index];

				if (goes_left) {
					children_left[dimension][children_left_count[dimension]++] = indices[dimension][i];
				} else {
					children_right[dimension][children_right_count[dimension]++] = indices[dimension][i];
				}
			}
		}

		// We should have made the same decision (going left/right) in every dimension
		assert(children_left_count [0] == children_left_count [1] && children_left_count [1] == children_left_count [2]);
		assert(children_right_count[0] == children_right_count[1] && children_right_count[1] == children_right_count[2]);

		n_left  = children_left_count [0];
		n_right = children_right_count[0];

		// Using object split, no duplicates can occur.
		// Thus, left + right should equal the total number of triangles
		assert(first_index + n_left == object_split.index);
		assert(n_left + n_right == index_count);

		child_aabb_left  = object_split.aabb_left;
		child_aabb_right = object_split.aabb_right;
	} else {
		// Perform Spatial Split

		node.count = 0;
		node.axis  = spatial_split.dimension;

		int temp_left = 0, temp_right = 0;

		// Keep track of amount of rejected references on both sides for debugging purposes
		int rejected_left  = 0;
		int rejected_right = 0;

		float n_1 = float(spatial_split.num_left);
		float n_2 = float(spatial_split.num_right);

		float bounds_min  = node.aabb.min[spatial_split.dimension] - 0.001f;
		float bounds_max  = node.aabb.max[spatial_split.dimension] + 0.001f;
		float bounds_step = (bounds_max - bounds_min) / BVHPartitions::SBVH_BIN_COUNT;

		float inv_bounds_delta = 1.0f / (bounds_max - bounds_min);

		for (int i = first_index; i < first_index + index_count; i++) {
			int index = indices[spatial_split.dimension][i].index;
			const Triangle & triangle = triangles[index];

			AABB triangle_aabb = indices[spatial_split.dimension][i].aabb;

			Vector3 vertices[3] = {
				triangle.position_0,
				triangle.position_1,
				triangle.position_2
			};

			// Sort the vertices along the current dimension
			if (vertices[0][spatial_split.dimension] > vertices[1][spatial_split.dimension]) Util::swap(vertices[0], vertices[1]);
			if (vertices[1][spatial_split.dimension] > vertices[2][spatial_split.dimension]) Util::swap(vertices[1], vertices[2]);
			if (vertices[0][spatial_split.dimension] > vertices[1][spatial_split.dimension]) Util::swap(vertices[0], vertices[1]);

			float vertex_min = triangle_aabb.min[spatial_split.dimension];
			float vertex_max = triangle_aabb.max[spatial_split.dimension];

			int bin_min = int(BVHPartitions::SBVH_BIN_COUNT * ((vertex_min - bounds_min) * inv_bounds_delta));
			int bin_max = int(BVHPartitions::SBVH_BIN_COUNT * ((vertex_max - bounds_min) * inv_bounds_delta));

			bool goes_left  = bin_min <  spatial_split.index;
			bool goes_right = bin_max >= spatial_split.index;

			assert(goes_left || goes_right);

			if (goes_left && goes_right) { // Straddler
				// Consider unsplitting
				AABB delta_left  = spatial_split.aabb_left;
				AABB delta_right = spatial_split.aabb_right;

				delta_left .expand(triangle_aabb);
				delta_right.expand(triangle_aabb);

				float spatial_split_aabb_left_surface_area  = spatial_split.aabb_left .surface_area();
				float spatial_split_aabb_right_surface_area = spatial_split.aabb_right.surface_area();

				// Calculate SAH cost for the 3 different cases
				float cost_split = spatial_split_aabb_left_surface_area   *  n_1       + spatial_split_aabb_right_surface_area   *  n_2;
				float cost_left  =              delta_left.surface_area() *  n_1       + spatial_split_aabb_right_surface_area   * (n_2-1.0f);
				float cost_right = spatial_split_aabb_left_surface_area   * (n_1-1.0f) +              delta_right.surface_area() *  n_2;

				// If cost_left resp. cost_right is cheapest, let the triangle go left resp. right
				// Otherwise, do nothing and let the triangle go both left and right
				if (cost_left < cost_split) {
					if (cost_right < cost_left) { // cost_right is cheapest, remove from left
						goes_left = false;
						rejected_left++;

						n_1 -= 1.0f;

						spatial_split.aabb_right.expand(triangle_aabb);
					} else { // cost_left is cheapest, remove from right
						goes_right = false;
						rejected_right++;

						n_2 -= 1.0f;

						spatial_split.aabb_left.expand(triangle_aabb);
					}
				} else if (cost_right < cost_split) { // cost_right is cheapest, remove from left
					goes_left = false;
					rejected_left++;

					n_1 -= 1.0f;

					spatial_split.aabb_right.expand(triangle_aabb);
				}
			}

			if (goes_left && goes_right) {
				Vector3 intersections[6];
				int     intersection_count = 0;

				BVHPartitions::triangle_intersect_plane(vertices, spatial_split.dimension, spatial_split.plane_distance, intersections, &intersection_count);

				assert(intersection_count < Util::array_element_count(intersections));

				// All intersection points should be included both AABBs
				AABB aabb_intersections = AABB::from_points(intersections, intersection_count);
				AABB aabb_left 	= aabb_intersections;
				AABB aabb_right = aabb_intersections;

				for (int v = 0; v < 3; v++) {
					if (vertices[v][spatial_split.dimension] < spatial_split.plane_distance) {
						aabb_left.expand(vertices[v]);
					} else {
						aabb_right.expand(vertices[v]);
					}
				}

				aabb_left.min  = Vector3::max(aabb_left.min,  triangle_aabb.min);
				aabb_left.max  = Vector3::min(aabb_left.max,  triangle_aabb.max);
				aabb_right.min = Vector3::max(aabb_right.min, triangle_aabb.min);
				aabb_right.max = Vector3::min(aabb_right.max, triangle_aabb.max);

				aabb_left .fix_if_needed();
				aabb_right.fix_if_needed();

				spatial_split.aabb_left .expand(aabb_left);
				spatial_split.aabb_right.expand(aabb_right);

				children_left[0][children_left_count[0]++] = { indices[spatial_split.dimension][i].index, aabb_left };
				children_left[1][children_left_count[1]++] = { indices[spatial_split.dimension][i].index, aabb_left };
				children_left[2][children_left_count[2]++] = { indices[spatial_split.dimension][i].index, aabb_left };

				children_right[0][children_right_count[0]++] = { indices[spatial_split.dimension][i].index, aabb_right };
				children_right[1][children_right_count[1]++] = { indices[spatial_split.dimension][i].index, aabb_right };
				children_right[2][children_right_count[2]++] = { indices[spatial_split.dimension][i].index, aabb_right };
			} else if (goes_left) {
				spatial_split.aabb_left.expand(triangle_aabb);

				children_left[0][children_left_count[0]++] = indices[spatial_split.dimension][i];
				children_left[1][children_left_count[1]++] = indices[spatial_split.dimension][i];
				children_left[2][children_left_count[2]++] = indices[spatial_split.dimension][i];
			} else if (goes_right) {
				spatial_split.aabb_right.expand(triangle_aabb);

				children_right[0][children_right_count[0]++] = indices[spatial_split.dimension][i];
				children_right[1][children_right_count[1]++] = indices[spatial_split.dimension][i];
				children_right[2][children_right_count[2]++] = indices[spatial_split.dimension][i];
			} else {
				assert(false);
			}
		}

		Util::quick_sort(children_left[0], children_left[0] + children_left_count[0], [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().x < b.aabb.get_center().x; });
		Util::quick_sort(children_left[1], children_left[1] + children_left_count[1], [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().y < b.aabb.get_center().y; });
		Util::quick_sort(children_left[2], children_left[2] + children_left_count[2], [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().z < b.aabb.get_center().z; });

		Util::quick_sort(children_right[0], children_right[0] + children_right_count[0], [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().x < b.aabb.get_center().x; });
		Util::quick_sort(children_right[1], children_right[1] + children_right_count[1], [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().y < b.aabb.get_center().y; });
		Util::quick_sort(children_right[2], children_right[2] + children_right_count[2], [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().z < b.aabb.get_center().z; });

		// We should have made the same decision (going left/right) in every dimension
		assert(children_left_count [0] == children_left_count [1] && children_left_count [1] == children_left_count [2]);
		assert(children_right_count[0] == children_right_count[1] && children_right_count[1] == children_right_count[2]);

		n_left  = children_left_count [0];
		n_right = children_right_count[0];

		// The actual number of references going left/right should match the numbers calculated during spatial splitting
		assert(n_left  == spatial_split.num_left  - rejected_left);
		assert(n_right == spatial_split.num_right - rejected_right);

		// A valid partition contains at least one and strictly less than all
		assert(n_left  > 0 && n_left  < index_count);
		assert(n_right > 0 && n_right < index_count);

		// Make sure no triangles dissapeared
		assert(n_left + n_right >= index_count);
		assert(n_left + n_right <= index_count * 2);

		child_aabb_left  = spatial_split.aabb_left;
		child_aabb_right = spatial_split.aabb_right;
	}

	sbvh->nodes_2[node.left    ].aabb = child_aabb_left;
	sbvh->nodes_2[node.left + 1].aabb = child_aabb_right;

	indices_going_right_offset += n_right;

	// Do a depth first traversal, so that we know the amount of indices that were recursively created by the left child
	int num_leaves_left = build_sbvh(sbvh->nodes_2[node.left], triangles, indices, node_index, first_index, n_left, inv_root_surface_area);

	// Using the depth first offset, we can now copy over the right references
	memcpy(indices[0] + first_index + num_leaves_left, children_right[0], n_right * sizeof(PrimitiveRef));
	memcpy(indices[1] + first_index + num_leaves_left, children_right[1], n_right * sizeof(PrimitiveRef));
	memcpy(indices[2] + first_index + num_leaves_left, children_right[2], n_right * sizeof(PrimitiveRef));

	// Now recurse on the right side
	int num_leaves_right = build_sbvh(sbvh->nodes_2[node.left + 1], triangles, indices, node_index, first_index + num_leaves_left, n_right, inv_root_surface_area);

	indices_going_right_offset -= n_right;

	return num_leaves_left + num_leaves_right;
}

void SBVHBuilder::init(BVH * sbvh, int triangle_count, int max_primitives_in_leaf) {
	this->sbvh = sbvh;
	this->max_primitives_in_leaf = max_primitives_in_leaf;

	PrimitiveRef * indices_xyz = new PrimitiveRef[3 * SBVH_OVERALLOCATION * triangle_count];
	indices_x = indices_xyz;
	indices_y = indices_xyz + SBVH_OVERALLOCATION * triangle_count;
	indices_z = indices_xyz + SBVH_OVERALLOCATION * triangle_count * 2;

	PrimitiveRef * indices_going_right_xyz = new PrimitiveRef[3 * SBVH_OVERALLOCATION * triangle_count];
	indices_going_right_x = indices_going_right_xyz;
	indices_going_right_y = indices_going_right_xyz + SBVH_OVERALLOCATION * triangle_count;
	indices_going_right_z = indices_going_right_xyz + SBVH_OVERALLOCATION * triangle_count * 2;

	sah    = new float[triangle_count];
	bounds = new AABB [triangle_count * 2 + 1];
	indices_going_left .init(triangle_count);
	indices_going_right.init(triangle_count);

	sbvh->indices = new int     [SBVH_OVERALLOCATION * triangle_count];
	sbvh->nodes_2 = new BVHNode2[SBVH_OVERALLOCATION * triangle_count];
}

void SBVHBuilder::free() {
	delete [] indices_x;
	delete [] indices_going_right_x;

	delete [] sah;
	delete [] bounds;
	indices_going_left .free();
	indices_going_right.free();
}

void SBVHBuilder::build(const Triangle * triangles, int triangle_count) {
	puts("Construcing SBVH, this may take a few seconds for large scenes...");

	AABB root_aabb = AABB::create_empty();

	for (int i = 0; i < triangle_count; i++) {
		indices_x[i].index = i;
		indices_y[i].index = i;
		indices_z[i].index = i;

		Vector3 vertices[3] = {
			triangles[i].position_0,
			triangles[i].position_1,
			triangles[i].position_2
		};
		AABB aabb = AABB::from_points(vertices, 3);

		indices_x[i].aabb = aabb;
		indices_y[i].aabb = aabb;
		indices_z[i].aabb = aabb;

		root_aabb.expand(aabb);
	}

	Util::quick_sort(indices_x, indices_x + triangle_count, [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().x < b.aabb.get_center().x; });
	Util::quick_sort(indices_y, indices_y + triangle_count, [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().y < b.aabb.get_center().y; });
	Util::quick_sort(indices_z, indices_z + triangle_count, [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().z < b.aabb.get_center().z; });

	indices_going_right_offset = 0;

	PrimitiveRef * indices[3] = { indices_x, indices_y, indices_z };

	sbvh->nodes_2[0].aabb = root_aabb;

	int node_index = 2;
	sbvh->index_count = build_sbvh(sbvh->nodes_2[0], triangles, indices, node_index, 0, triangle_count, 1.0f / root_aabb.surface_area());
	sbvh->node_count = node_index;

	assert(node_index <= SBVH_OVERALLOCATION * triangle_count);

	for (int i = 0; i < sbvh->index_count; i++) {
		int index = indices_x[i].index;
		assert(index >= 0 && index < triangle_count);

		sbvh->indices[i] = index;
	}
}
