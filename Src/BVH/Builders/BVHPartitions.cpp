#include "BVHPartitions.h"

#include "Renderer/Mesh.h"
#include "Renderer/Triangle.h"

// Evaluates SAH for every object for every dimension to determine splitting candidate
template<typename GetAABB>
ObjectSplit partition_sah_impl(GetAABB get_aabb, int first_index, int index_count, float * sah) {
	ObjectSplit split = { };
	split.cost = INFINITY;
	split.index     = -1;
	split.dimension = -1;
	split.aabb_left  = AABB::create_empty();
	split.aabb_right = AABB::create_empty();

	// Check splits along all 3 dimensions
	for (int dimension = 0; dimension < 3; dimension++) {
		AABB aabb_left  = AABB::create_empty();
		AABB aabb_right = AABB::create_empty();

		// First traverse left to right along the current dimension to evaluate first half of the SAH
		for (int i = 1; i < index_count; i++) {
			aabb_left.expand(get_aabb(dimension, first_index + i - 1));

			sah[i] = aabb_left.surface_area() * float(i);
		}

		// Then traverse right to left along the current dimension to evaluate second half of the SAH
		for (int i = index_count - 1; i > 0; i--) {
			aabb_right.expand(get_aabb(dimension, first_index + i));

			float cost = sah[i] + aabb_right.surface_area() * float(index_count - i);
			if (cost <= split.cost) {
				split.cost = cost;
				split.index = first_index + i;
				split.dimension = dimension;
				split.aabb_right = aabb_right;
			}
		}

		ASSERT(!aabb_left .is_empty());
		ASSERT(!aabb_right.is_empty());
	}

	// Calculate left AABB, right AABB was already calculated above
	for (int i = first_index; i < split.index; i++) {
		split.aabb_left.expand(get_aabb(split.dimension, i));
	}

	return split;
}

ObjectSplit BVHPartitions::partition_sah(const Array<Triangle> & triangles, int * indices[3], int first_index, int index_count, float * sah) {
	auto get_aabb = [&triangles, &indices](int dimension, int index) {
		return triangles[indices[dimension][index]].aabb;
	};
	return partition_sah_impl(get_aabb, first_index, index_count, sah);
}

ObjectSplit BVHPartitions::partition_sah(const Array<Mesh> & meshes, int * indices[3], int first_index, int index_count, float * sah) {
	auto get_aabb = [&meshes, &indices](int dimension, int index) {
		return meshes[indices[dimension][index]].aabb;
	};
	return partition_sah_impl(get_aabb, first_index, index_count, sah);
}

ObjectSplit BVHPartitions::partition_sah(Array<PrimitiveRef> primitive_refs[3], int first_index, int index_count, float * sah) {
	auto get_aabb = [&primitive_refs](int dimension, int index) {
		return primitive_refs[dimension][index].aabb;
	};
	return partition_sah_impl(get_aabb, first_index, index_count, sah);
}

void BVHPartitions::triangle_intersect_plane(Vector3 vertices[3], int dimension, float plane, Vector3 intersections[], int * intersection_count) {
	for (int i = 0; i < 3; i++) {
		float vertex_i = vertices[i][dimension];

		for (int j = i + 1; j < 3; j++) {
			float vertex_j = vertices[j][dimension];

			float delta_ij = vertex_j - vertex_i;

			// Check if edge between Vertex i and j intersects the plane
			if (vertex_i <= plane && plane <= vertex_j) {
				if (delta_ij == 0) {
					intersections[(*intersection_count)++] = vertices[i];
					intersections[(*intersection_count)++] = vertices[j];
				} else {
					// Lerp to obtain exact intersection point
					float t = (plane - vertex_i) / delta_ij;
					intersections[(*intersection_count)++] = (1.0f - t) * vertices[i] + t * vertices[j];
				}
			}
		}
	}
}

SpatialSplit BVHPartitions::partition_spatial(const Array<Triangle> & triangles, const Array<PrimitiveRef> indices[3], int first_index, int index_count, float * sah, AABB bounds) {
	SpatialSplit split = { };
	split.cost = INFINITY;
	split.index     = -1;
	split.dimension = -1;
	split.plane_distance = NAN;

	for (int dimension = 0; dimension < 3; dimension++) {
		float bounds_min  = bounds.min[dimension] - 0.001f;
		float bounds_max  = bounds.max[dimension] + 0.001f;
		float bounds_step = (bounds_max - bounds_min) / SBVH_BIN_COUNT;

		float inv_bounds_delta = 1.0f / (bounds_max - bounds_min);

		struct Bin {
			AABB aabb = AABB::create_empty();
			int entries = 0;
			int exits   = 0;
		} bins[SBVH_BIN_COUNT];

		for (int i = first_index; i < first_index + index_count; i++) {
			const Triangle & triangle = triangles[indices[dimension][i].index];

			AABB triangle_aabb = indices[dimension][i].aabb;

			Vector3 vertices[3] = {
				triangle.position_0,
				triangle.position_1,
				triangle.position_2
			};

			// Sort the vertices along the current dimension
			if (vertices[0][dimension] > vertices[1][dimension]) Util::swap(vertices[0], vertices[1]);
			if (vertices[1][dimension] > vertices[2][dimension]) Util::swap(vertices[1], vertices[2]);
			if (vertices[0][dimension] > vertices[1][dimension]) Util::swap(vertices[0], vertices[1]);

			float triangle_aabb_min = triangle_aabb.min[dimension];
			float triangle_aabb_max = triangle_aabb.max[dimension];

			int bin_min = int(SBVH_BIN_COUNT * ((triangle_aabb_min - bounds_min) * inv_bounds_delta));
			int bin_max = int(SBVH_BIN_COUNT * ((triangle_aabb_max - bounds_min) * inv_bounds_delta));

			bin_min = Math::clamp(bin_min, 0, SBVH_BIN_COUNT - 1);
			bin_max = Math::clamp(bin_max, 0, SBVH_BIN_COUNT - 1);

			bins[bin_min].entries++;
			bins[bin_max].exits++;

			// Iterate over bins that intersect the AABB along the current dimension
			for (int b = bin_min; b <= bin_max; b++) {
				Bin & bin = bins[b];

				float bin_left_plane  = bounds_min + float(b) * bounds_step;
				float bin_right_plane = bin_left_plane + bounds_step;

				ASSERT(bin.aabb.is_valid() || bin.aabb.is_empty());

				// If all vertices lie outside the bin we don't care about this triangle
				if (triangle_aabb_min >= bin_right_plane || triangle_aabb_max <= bin_left_plane) {
					continue;
				}

				// Calculate relevant portion of the AABB with regard to the two planes that define the current Bin
				AABB triangle_aabb_clipped_against_bin = AABB::create_empty();

				// If all verticies lie between the two planes, the AABB is just the Triangle's entire AABB
				if (triangle_aabb_min >= bin_left_plane && triangle_aabb_max <= bin_right_plane) {
					triangle_aabb_clipped_against_bin = triangle_aabb;
				} else {
					Vector3 intersections[12];
					int     intersection_count = 0;

					if (triangle_aabb_min <= bin_left_plane  && bin_left_plane  <= triangle_aabb_max) triangle_intersect_plane(vertices, dimension, bin_left_plane,  intersections, &intersection_count);
					if (triangle_aabb_min <= bin_right_plane && bin_right_plane <= triangle_aabb_max) triangle_intersect_plane(vertices, dimension, bin_right_plane, intersections, &intersection_count);

					ASSERT(intersection_count < Util::array_count(intersections));

					if (intersection_count == 0) {
						triangle_aabb_clipped_against_bin = triangle_aabb;
					} else {
						// All intersection points should be included in the AABB
						triangle_aabb_clipped_against_bin = AABB::from_points(intersections, intersection_count);

						// If the middle vertex lies between the two planes it should be included in the AABB
						if (vertices[1][dimension] >= bin_left_plane && vertices[1][dimension] < bin_right_plane) {
							triangle_aabb_clipped_against_bin.expand(vertices[1]);
						}

						if (vertices[2][dimension] <= bin_right_plane && vertices[2][dimension] <= triangle_aabb_max) triangle_aabb_clipped_against_bin.expand(vertices[2]);
						if (vertices[0][dimension] >= bin_left_plane  && vertices[0][dimension] >= triangle_aabb_min) triangle_aabb_clipped_against_bin.expand(vertices[0]);

						triangle_aabb_clipped_against_bin = AABB::overlap(triangle_aabb_clipped_against_bin, triangle_aabb);
					}
				}

				// Clip the AABB against the parent bounds
				bin.aabb.expand(triangle_aabb_clipped_against_bin);
				bin.aabb = AABB::overlap(bin.aabb, bounds);

				bin.aabb.fix_if_needed();

				// AABB must be valid
				ASSERT(bin.aabb.is_valid() || bin.aabb.is_empty());

				// The AABB of the current Bin cannot exceed the planes of the current Bin
				const float epsilon = 0.01f;
				ASSERT(bin.aabb.min[dimension] > bin_left_plane  - epsilon);
				ASSERT(bin.aabb.max[dimension] < bin_right_plane + epsilon);

				// The AABB of the current Bin cannot exceed the bounds of the Node's AABB
				ASSERT(bin.aabb.min[0] > bounds.min[0] - epsilon && bin.aabb.max[0] < bounds.max[0] + epsilon);
				ASSERT(bin.aabb.min[1] > bounds.min[1] - epsilon && bin.aabb.max[1] < bounds.max[1] + epsilon);
				ASSERT(bin.aabb.min[2] > bounds.min[2] - epsilon && bin.aabb.max[2] < bounds.max[2] + epsilon);
			}
		}

		float bin_sah[SBVH_BIN_COUNT];

		AABB bounds_left [SBVH_BIN_COUNT];
		AABB bounds_right[SBVH_BIN_COUNT + 1];

		bounds_left [0]              = AABB::create_empty();
		bounds_right[SBVH_BIN_COUNT] = AABB::create_empty();

		int count_left [SBVH_BIN_COUNT];
		int count_right[SBVH_BIN_COUNT + 1];

		count_left [0]              = 0;
		count_right[SBVH_BIN_COUNT] = 0;

		// First traverse left to right along the current dimension to evaluate first half of the SAH
		for (int b = 1; b < SBVH_BIN_COUNT; b++) {
			bounds_left[b] = bounds_left[b-1];
			bounds_left[b].expand(bins[b-1].aabb);

			ASSERT(bounds_left[b].is_valid() || bounds_left[b].is_empty());

			count_left[b] = count_left[b-1] + bins[b-1].entries;

			if (count_left[b] < index_count) {
				bin_sah[b] = bounds_left[b].surface_area() * float(count_left[b]);
			} else {
				bin_sah[b] = INFINITY;
			}
		}

		// Then traverse right to left along the current dimension to evaluate second half of the SAH
		for (int b = SBVH_BIN_COUNT - 1; b > 0; b--) {
			bounds_right[b] = bounds_right[b+1];
			bounds_right[b].expand(bins[b].aabb);

			ASSERT(bounds_right[b].is_valid() || bounds_right[b].is_empty());

			count_right[b] = count_right[b+1] + bins[b].exits;

			if (count_right[b] < index_count) {
				bin_sah[b] += bounds_right[b].surface_area() * float(count_right[b]);
			} else {
				bin_sah[b] = INFINITY;
			}
		}

		ASSERT(count_left [SBVH_BIN_COUNT - 1] + bins[SBVH_BIN_COUNT - 1].entries == index_count);
		ASSERT(count_right[1]                  + bins[0].exits                    == index_count);

		// Find the splitting plane that yields the lowest SAH cost along the current dimension
		for (int b = 1; b < SBVH_BIN_COUNT; b++) {
			float cost = bin_sah[b];
			if (cost < split.cost) {
				split.cost = cost;
				split.index = b;
				split.dimension = dimension;

				split.plane_distance = bounds_min + bounds_step * float(b);

				split.aabb_left  = bounds_left [b];
				split.aabb_right = bounds_right[b];

				split.num_left  = count_left [b];
				split.num_right = count_right[b];
			}
		}
	}

	return split;
}