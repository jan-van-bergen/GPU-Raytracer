#include "SBVHBuilder.h"

#include "Config.h"

#include "Core/IO.h"
#include "Core/Assertion.h"
#include "Core/ScopeTimer.h"

#include "BVHPartitions.h"

#include "Util/Util.h"

void SBVHBuilder::build(const Array<Triangle> & triangles) {
	IO::print("Construcing SBVH, this may take a few seconds for large Meshes...\n"_sv);

	AABB root_aabb = AABB::create_empty();

	indices[0].resize(triangles.size());
	indices[1].resize(triangles.size());
	indices[2].resize(triangles.size());

	for (size_t i = 0; i < triangles.size(); i++) {
		indices[0][i].index = i;
		indices[1][i].index = i;
		indices[2][i].index = i;

		Vector3 vertices[3] = {
			triangles[i].position_0,
			triangles[i].position_1,
			triangles[i].position_2
		};
		AABB aabb = AABB::from_points(vertices, 3);

		indices[0][i].aabb = aabb;
		indices[1][i].aabb = aabb;
		indices[2][i].aabb = aabb;

		root_aabb.expand(aabb);
	}

	inv_root_surface_area = 1.0f / root_aabb.surface_area();

	Util::quick_sort(indices[0].begin(), indices[0].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().x < b.aabb.get_center().x; });
	Util::quick_sort(indices[1].begin(), indices[1].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().y < b.aabb.get_center().y; });
	Util::quick_sort(indices[2].begin(), indices[2].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().z < b.aabb.get_center().z; });

	sbvh.nodes.clear();
	sbvh.nodes.reserve(2 * triangles.size());
	sbvh.nodes.emplace_back(); // Root
	sbvh.nodes.emplace_back(); // Dummy
	sbvh.nodes[0].aabb = root_aabb;

	int index_count = build_sbvh(0, triangles, 0, triangles.size());

	sbvh.indices.resize(index_count);
	for (int i = 0; i < index_count; i++) {
		int index = indices[0][i].index;
		ASSERT(index >= 0 && index < triangles.size());

		sbvh.indices[i] = index;
	}
}

int SBVHBuilder::build_sbvh(int node_index, const Array<Triangle> & triangles, int first_index, int index_count) {
	if (index_count == 1) {
		// Leaf Node, terminate recursion
		// We do not terminate based on the SAH termination criterion, so that the
		// BVHs that are cached to disk have a standard layout (1 triangle per leaf node)
		// If desired these trees can be collapsed based on the SAH cost using BVHCollapser::collapse
		sbvh.nodes[node_index].first = first_index;
		sbvh.nodes[node_index].count = index_count;

		return index_count;
	}

	// Object Split information
	ObjectSplit object_split = BVHPartitions::partition_sah(indices, first_index, index_count, sah.data());
	ASSERT(object_split.index != INVALID);

	// Calculate the overlap between the child bounding boxes resulting from the Object Split
	AABB overlap = AABB::overlap(object_split.aabb_left, object_split.aabb_right);
	float lamba = overlap.is_valid() ? overlap.surface_area() : 0.0f;

	// Divide by the surface area of the bounding box of the root Node
	float ratio = lamba * inv_root_surface_area;
	ASSERT(ratio >= 0.0f && ratio <= 1.0f);

	SpatialSplit spatial_split;

	// If ratio between overlap area and root area is large enough, consider a Spatial Split
	if (ratio > config.sbvh_alpha) {
		spatial_split = BVHPartitions::partition_spatial(triangles, indices, first_index, index_count, sah.data(), sbvh.nodes[node_index].aabb);
	} else {
		spatial_split.cost = INFINITY;
	}

	ASSERT(isfinite(object_split.cost) || isfinite(spatial_split.cost));

	sbvh.nodes[node_index].left = sbvh.nodes.size();
	sbvh.nodes.emplace_back(); // Left child
	sbvh.nodes.emplace_back(); // Right child

	Array<PrimitiveRef> children_left [3];
	Array<PrimitiveRef> children_right[3];
	for (int i = 0; i < 3; i++) {
		children_left [i].reserve(index_count);
		children_right[i].reserve(index_count);
	}

	int n_left, n_right;

	AABB child_aabb_left;
	AABB child_aabb_right;

	// The two temp arrays will be used as lookup tables
	if (object_split.cost <= spatial_split.cost) {
		// Perform Object Split

		sbvh.nodes[node_index].count = 0;
		sbvh.nodes[node_index].axis  = object_split.dimension;

		for (int i = first_index;        i < object_split.index;        i++) indices_going_left[indices[object_split.dimension][i].index] = true;
		for (int i = object_split.index; i < first_index + index_count; i++) indices_going_left[indices[object_split.dimension][i].index] = false;

		for (int dimension = 0; dimension < 3; dimension++) {
			for (int i = first_index; i < first_index + index_count; i++) {
				bool goes_left = indices_going_left[indices[dimension][i].index];

				if (goes_left) {
					children_left[dimension].push_back(indices[dimension][i]);
				} else {
					children_right[dimension].push_back(indices[dimension][i]);
				}
			}
		}

		// We should have made the same decision (going left/right) in every dimension
		ASSERT(children_left [0].size() == children_left [1].size() && children_left [1].size() == children_left [2].size());
		ASSERT(children_right[0].size() == children_right[1].size() && children_right[1].size() == children_right[2].size());

		n_left  = children_left [0].size();
		n_right = children_right[0].size();

		// Using object split, no duplicates can occur.
		// Thus, left + right should equal the total number of triangles
		ASSERT(first_index + n_left == object_split.index);
		ASSERT(n_left + n_right == index_count);

		child_aabb_left  = object_split.aabb_left;
		child_aabb_right = object_split.aabb_right;
	} else {
		// Perform Spatial Split

		sbvh.nodes[node_index].count = 0;
		sbvh.nodes[node_index].axis  = spatial_split.dimension;

		// Keep track of amount of rejected references on both sides for debugging purposes
		int rejected_left  = 0;
		int rejected_right = 0;

		float n_1 = float(spatial_split.num_left);
		float n_2 = float(spatial_split.num_right);

		float bounds_min  = sbvh.nodes[node_index].aabb.min[spatial_split.dimension] - 0.001f;
		float bounds_max  = sbvh.nodes[node_index].aabb.max[spatial_split.dimension] + 0.001f;

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

			ASSERT(goes_left || goes_right);

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

				ASSERT(intersection_count < Util::array_count(intersections));

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

				children_left[0].emplace_back(indices[spatial_split.dimension][i].index, aabb_left);
				children_left[1].emplace_back(indices[spatial_split.dimension][i].index, aabb_left);
				children_left[2].emplace_back(indices[spatial_split.dimension][i].index, aabb_left);

				children_right[0].emplace_back(indices[spatial_split.dimension][i].index, aabb_right);
				children_right[1].emplace_back(indices[spatial_split.dimension][i].index, aabb_right);
				children_right[2].emplace_back(indices[spatial_split.dimension][i].index, aabb_right);
			} else if (goes_left) {
				spatial_split.aabb_left.expand(triangle_aabb);

				children_left[0].push_back(indices[spatial_split.dimension][i]);
				children_left[1].push_back(indices[spatial_split.dimension][i]);
				children_left[2].push_back(indices[spatial_split.dimension][i]);
			} else if (goes_right) {
				spatial_split.aabb_right.expand(triangle_aabb);

				children_right[0].push_back(indices[spatial_split.dimension][i]);
				children_right[1].push_back(indices[spatial_split.dimension][i]);
				children_right[2].push_back(indices[spatial_split.dimension][i]);
			} else {
				ASSERT(false);
			}
		}

		Util::quick_sort(children_left[0].begin(), children_left[0].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().x < b.aabb.get_center().x; });
		Util::quick_sort(children_left[1].begin(), children_left[1].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().y < b.aabb.get_center().y; });
		Util::quick_sort(children_left[2].begin(), children_left[2].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().z < b.aabb.get_center().z; });

		Util::quick_sort(children_right[0].begin(), children_right[0].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().x < b.aabb.get_center().x; });
		Util::quick_sort(children_right[1].begin(), children_right[1].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().y < b.aabb.get_center().y; });
		Util::quick_sort(children_right[2].begin(), children_right[2].end(), [](const PrimitiveRef & a, const PrimitiveRef & b) { return a.aabb.get_center().z < b.aabb.get_center().z; });

		// We should have made the same decision (going left/right) in every dimension
		ASSERT(children_left [0].size() == children_left [1].size() && children_left [1].size() == children_left [2].size());
		ASSERT(children_right[0].size() == children_right[1].size() && children_right[1].size() == children_right[2].size());

		n_left  = children_left [0].size();
		n_right = children_right[0].size();

		// The actual number of references going left/right should match the numbers calculated during spatial splitting
		ASSERT(n_left  == spatial_split.num_left  - rejected_left);
		ASSERT(n_right == spatial_split.num_right - rejected_right);

		// A valid partition contains at least one and strictly less than all
		ASSERT(n_left  > 0 && n_left  < index_count);
		ASSERT(n_right > 0 && n_right < index_count);

		// Make sure no triangles dissapeared
		ASSERT(n_left + n_right >= index_count);
		ASSERT(n_left + n_right <= index_count * 2);

		child_aabb_left  = spatial_split.aabb_left;
		child_aabb_right = spatial_split.aabb_right;
	}

	sbvh.nodes[sbvh.nodes[node_index].left    ].aabb = child_aabb_left;
	sbvh.nodes[sbvh.nodes[node_index].left + 1].aabb = child_aabb_right;

	indices[0].resize_if_smaller(first_index + n_left);
	indices[1].resize_if_smaller(first_index + n_left);
	indices[2].resize_if_smaller(first_index + n_left);

	memcpy(indices[0].data() + first_index, children_left[0].data(), n_left * sizeof(PrimitiveRef));
	memcpy(indices[1].data() + first_index, children_left[1].data(), n_left * sizeof(PrimitiveRef));
	memcpy(indices[2].data() + first_index, children_left[2].data(), n_left * sizeof(PrimitiveRef));

	children_left[0] = { };
	children_left[1] = { };
	children_left[2] = { };

	// Do a depth first traversal, so that we know the amount of indices that were recursively created by the left child
	int num_leaves_left = build_sbvh(sbvh.nodes[node_index].left, triangles, first_index, n_left);

	indices[0].resize_if_smaller(first_index + num_leaves_left + n_right);
	indices[1].resize_if_smaller(first_index + num_leaves_left + n_right);
	indices[2].resize_if_smaller(first_index + num_leaves_left + n_right);

	// Using the depth first offset, we can now copy over the right references
	memcpy(indices[0].data() + first_index + num_leaves_left, children_right[0].data(), n_right * sizeof(PrimitiveRef));
	memcpy(indices[1].data() + first_index + num_leaves_left, children_right[1].data(), n_right * sizeof(PrimitiveRef));
	memcpy(indices[2].data() + first_index + num_leaves_left, children_right[2].data(), n_right * sizeof(PrimitiveRef));

	// Now recurse on the right side
	int num_leaves_right = build_sbvh(sbvh.nodes[node_index].left + 1, triangles, first_index + num_leaves_left, n_right);

	return num_leaves_left + num_leaves_right;
}
