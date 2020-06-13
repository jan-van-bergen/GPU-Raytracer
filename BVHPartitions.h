#pragma once
#include "Math.h"

#include "Util.h"

// Contains various ways to parition space into "left" and "right" as well as helper methods
namespace BVHPartitions {
	const int SBVH_BIN_COUNT = 256;
		
	// Calculates the smallest enclosing AABB over the union of all AABB's of the primitives in the range defined by [first, last>
	template<typename PrimitiveType>
	inline AABB calculate_bounds(const PrimitiveType * primitives, const int * indices, int first, int last) {
		AABB aabb = AABB::create_empty();

		// Iterate over relevant Primitives
		for (int i = first; i < last; i++) {
			aabb.expand(primitives[indices[i]].aabb);
		}

		aabb.fix_if_needed();

		assert(aabb.is_valid());

		return aabb;
	} 

	// Used for debugging
	inline bool is_sorted(const Triangle * triangles, int * indices[3], int first, int last) {
		for (int dimension = 0; dimension < 3; dimension++) {
			for (int i = first + 1; i < last; i++) {
				float prev = triangles[indices[dimension][i-1]].get_center()[dimension];
				float curr = triangles[indices[dimension][i  ]].get_center()[dimension];

				if (prev > curr) return false;
			}
		}

		return true;
	}

	// Used for debugging
	inline bool is_unique(const Triangle * triangles, int * indices[3], int first_index, int index_count) {
		for (int dimension = 0; dimension < 3; dimension++) {
			for (int i = first_index; i < first_index + index_count; i++) {
				for (int j = first_index; j < i; j++) {
					if (indices[dimension][j] == indices[dimension][i]) return false;
				}
			}
		}

		return true;
	}

	// Reorders indices arrays such that indices on the left side of the splitting dimension end up on the left partition in the other dimensions as well
	inline void split_indices(const Triangle * triangles, int * indices[3], int first_index, int index_count, int * temp, int split_dimension, int split_index, float split) {
		for (int dimension = 0; dimension < 3; dimension++) {
			if (dimension != split_dimension) {
				int left  = 0;
				int right = split_index - first_index;

				for (int i = first_index; i < first_index + index_count; i++) {
					bool goes_left = triangles[indices[dimension][i]].get_center()[split_dimension] < split;

					if (triangles[indices[dimension][i]].get_center()[split_dimension] == split) {
						// In case the current primitive has the same coordianate as the one we split on along the split dimension,
						// We don't know whether the primitive should go left or right.
						// In this case check all primitive indices on the left side of the split that 
						// have the same split coordinate for equality with the current primitive index i

						int j = split_index - 1;
						// While we can go left and the left primitive has the same coordinate along the split dimension as the split itself
						while (j >= first_index && triangles[indices[split_dimension][j]].get_center()[split_dimension] == split) {
							if (indices[split_dimension][j] == indices[dimension][i]) {
								goes_left = true;

								break;
							}

							j--;
						}
					}

					if (goes_left) {					
						temp[left++] = indices[dimension][i];
					} else {
						temp[right++] = indices[dimension][i];
					}
				}

				// If these conditions are not met the memcpy below is invalid
				assert(left  == split_index - first_index);
				assert(right == index_count);

				memcpy(indices[dimension] + first_index, temp, index_count * sizeof(int));

				assert(is_sorted(triangles, indices, first_index,        left ));
				assert(is_sorted(triangles, indices, first_index + left, right));
			}
		}
	}

	// Evaluates SAH for every object for every dimension to determine splitting candidate
	inline int partition_sah(const Triangle * triangles, int * indices[3], int first_index, int index_count, float * sah, int & split_dimension, float & split_cost) {
		float min_split_cost = INFINITY;
		int   min_split_index     = -1;
		int   min_split_dimension = -1;

		// Check splits along all 3 dimensions
		for (int dimension = 0; dimension < 3; dimension++) {
			assert(is_sorted(triangles, indices, first_index, index_count));

			AABB aabb_left  = AABB::create_empty();
			AABB aabb_right = AABB::create_empty();

			// First traverse left to right along the current dimension to evaluate first half of the SAH
			for (int i = 0; i < index_count - 1; i++) {
				aabb_left.expand(triangles[indices[dimension][first_index + i]].aabb);
				
				sah[i] = aabb_left.surface_area() * float(i + 1);
			}

			// Then traverse right to left along the current dimension to evaluate second half of the SAH
			for (int i = index_count - 1; i > 0; i--) {
				aabb_right.expand(triangles[indices[dimension][first_index + i]].aabb);

				sah[i - 1] += aabb_right.surface_area() * float(index_count - i);
			}

			// Find the minimum of the SAH
			for (int i = 0; i < index_count - 1; i++) {
				float cost = sah[i];
				if (cost < min_split_cost) {
					min_split_cost = cost;
					min_split_index = first_index + i + 1;
					min_split_dimension = dimension;
				}
			}
		}
		
		split_cost      = min_split_cost;
		split_dimension = min_split_dimension;

		return min_split_index;
	}

	// Evaluates SAH for every object for every dimension to determine splitting candidate
	inline int partition_object(const Triangle * primitives, int * indices[3], int first_index, int index_count, float * sah, int & split_dimension, float & split_cost, const AABB & node_aabb, AABB & aabb_left, AABB & aabb_right) {
		float min_split_cost = INFINITY;
		int   min_split_index     = -1;
		int   min_split_dimension = -1;
		
		AABB * bounds       = new AABB[index_count + index_count + 1];
		AABB * bounds_left  = bounds;
		AABB * bounds_right = bounds + index_count;
		
		// Check splits along all 3 dimensions
		for (int dimension = 0; dimension < 3; dimension++) {
			assert(is_sorted(primitives, indices, first_index, index_count));

			bounds_left [0]           = AABB::create_empty();
			bounds_right[index_count] = AABB::create_empty();

			// First traverse left to right along the current dimension to evaluate first half of the SAH
			for (int i = 1; i < index_count; i++) {
				bounds_left[i] = bounds_left[i-1];
				bounds_left[i].expand(primitives[indices[dimension][first_index + i - 1]].aabb);
				bounds_left[i] = AABB::overlap(bounds_left[i], node_aabb);

				sah[i] = bounds_left[i].surface_area() * float(i);
			}

			// Then traverse right to left along the current dimension to evaluate second half of the SAH
			for (int i = index_count - 1; i > 0; i--) {
				bounds_right[i] = bounds_right[i+1];
				bounds_right[i].expand(primitives[indices[dimension][first_index + i]].aabb);
				bounds_right[i] = AABB::overlap(bounds_right[i], node_aabb);
				
				sah[i] += bounds_right[i].surface_area() * float(index_count - i);
			}
			
			// Find the minimum of the SAH
			for (int i = 1; i < index_count; i++) {
				float cost = sah[i];
				if (cost < min_split_cost) {
					min_split_cost = cost;
					min_split_index = first_index + i;
					min_split_dimension = dimension;
					
					assert(!bounds_left [i].is_empty());
					assert(!bounds_right[i].is_empty());

					aabb_left  = bounds_left[i];
					aabb_right = bounds_right[i];
				}
			}
		}

		delete [] bounds;
		
		split_dimension = min_split_dimension;
		split_cost      = min_split_cost;

		return min_split_index;
	}

	inline int partition_spatial(const Triangle * triangles, int * indices[3], int first_index, int index_count, float * sah, int & split_dimension, float & split_cost, AABB & aabb_left, AABB & aabb_right, int & n_left, int & n_right, AABB bounds) {
		float min_bin_cost = INFINITY;
		int   min_bin_index     = -1;
		int   min_bin_dimension = -1;
		float min_bin_plane_distance = NAN;

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
				const Triangle & triangle = triangles[indices[dimension][i]];
				
				AABB triangle_aabb = AABB::overlap(triangle.aabb, bounds);

				Vector3 vertices[3] = { 
					triangle.position_0,
					triangle.position_1, 
					triangle.position_2 
				};
				
				// Sort the vertices along the current dimension using unrolled Bubble Sort
				if (vertices[0][dimension] > vertices[1][dimension]) Util::swap(vertices[0], vertices[1]);
				if (vertices[1][dimension] > vertices[2][dimension]) Util::swap(vertices[1], vertices[2]);
				if (vertices[0][dimension] > vertices[1][dimension]) Util::swap(vertices[0], vertices[1]);

				float vertex_min = vertices[0][dimension];
				float vertex_max = vertices[2][dimension];
				
				int bin_min = int(SBVH_BIN_COUNT * ((triangle_aabb.min[dimension] - bounds_min) * inv_bounds_delta));
				int bin_max = int(SBVH_BIN_COUNT * ((triangle_aabb.max[dimension] - bounds_min) * inv_bounds_delta));

				bin_min = Math::clamp(bin_min, 0, SBVH_BIN_COUNT - 1);
				bin_max = Math::clamp(bin_max, 0, SBVH_BIN_COUNT - 1);

				bins[bin_min].entries++;
				bins[bin_max].exits++;

				// Iterate over bins that intersect the AABB along the current dimension
				for (int b = bin_min; b <= bin_max; b++) {
					Bin & bin = bins[b];
					
					float bin_left_plane  = bounds_min + float(b) * bounds_step;
					float bin_right_plane = bin_left_plane + bounds_step;

					assert(bin.aabb.is_valid() || bin.aabb.is_empty());

					// Calculate relevant portion of the AABB with regard to the two planes that define the current Bin
					AABB box;

					// If all vertices lie on one side of either plane the AABB is empty
					if (vertex_min >= bin_right_plane || vertex_max <= bin_left_plane) {
						continue;
					// If all verticies lie between the two planes, the AABB is just the Triangle's entire AABB
					} else if (vertex_min >= bin_left_plane && vertex_max <= bin_right_plane) {
						box = triangle_aabb;
					} else {
						Vector3 intersections[4];
						int     intersection_count = 0;

						for (int i = 0; i < 3; i++) {
							float vertex_i = vertices[i][dimension];

							for (int j = i + 1; j < 3; j++) {
								float vertex_j = vertices[j][dimension];

								float delta_ij = vertex_j - vertex_i;

								// Check if edge between Vertex i and j intersects the left plane
								if (vertex_i < bin_left_plane && bin_left_plane <= vertex_j) { 
									// Lerp to obtain exact intersection point
									float t = (bin_left_plane - vertex_i) / delta_ij;
									intersections[intersection_count++] = (1.0f - t) * vertices[i] + t * vertices[j];
								}

								// Check if edge between Vertex i and j intersects the right plane
								if (vertex_i < bin_right_plane && bin_right_plane <= vertex_j) { 
									// Lerp to obtain exact intersection point
									float t = (bin_right_plane - vertex_i) / delta_ij;
									intersections[intersection_count++] = (1.0f - t) * vertices[i] + t * vertices[j];
								}
							}
						}

						// There must be either 2 or 4 intersections with the two planes
						assert(intersection_count == 2 || intersection_count == 4);

						// All intersection points should be included in the AABB
						box = AABB::from_points(intersections, intersection_count);

						// If the middle vertex lies between the two planes it should be included in the AABB
						if (vertices[1][dimension] >= bin_left_plane && vertices[1][dimension] < bin_right_plane) {
							box.expand(vertices[1]);
						}

						// In case we have only two intersections with either plane it must be the case that
						// either the leftmost or the rightmost vertex lies between the two planes
						if (intersection_count == 2) {
							box.expand(vertex_max < bin_right_plane ? vertices[2] : vertices[0]);
						}

						box.fix_if_needed();
					}

					// Clip the AABB against the parent bounds
					bin.aabb.expand(box);
					bin.aabb = AABB::overlap(bin.aabb, bounds);

					// AABB must be valid
					assert(bin.aabb.is_valid() || bin.aabb.is_empty());
					
					// The AABB of the current Bin cannot exceed the planes of the current Bin
					const float epsilon = 0.01f;
					assert(bin.aabb.min[dimension] > bin_left_plane  - epsilon);
					assert(bin.aabb.max[dimension] < bin_right_plane + epsilon);

					// The AABB of the current Bin cannot exceed the bounds of the Node's AABB
					assert(bin.aabb.min[0] > bounds.min[0] - epsilon && bin.aabb.max[0] < bounds.max[0] + epsilon);
					assert(bin.aabb.min[1] > bounds.min[1] - epsilon && bin.aabb.max[1] < bounds.max[1] + epsilon);
					assert(bin.aabb.min[2] > bounds.min[2] - epsilon && bin.aabb.max[2] < bounds.max[2] + epsilon);
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

				assert(bounds_left[b].is_valid() || bounds_left[b].is_empty());

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
				
				assert(bounds_right[b].is_valid() || bounds_right[b].is_empty());

				count_right[b] = count_right[b+1] + bins[b].exits;

				if (count_right[b] < index_count) {
					bin_sah[b] += bounds_right[b].surface_area() * float(count_right[b]);
				} else {
					bin_sah[b] = INFINITY;
				}
			}

			assert(count_left [SBVH_BIN_COUNT - 1] + bins[SBVH_BIN_COUNT - 1].entries == index_count);
			assert(count_right[1]                  + bins[0].exits                    == index_count);

			// Find the splitting plane that yields the lowest SAH cost along the current dimension
			for (int b = 1; b < SBVH_BIN_COUNT; b++) {
				float cost = bin_sah[b];
				if (cost < min_bin_cost) {
					min_bin_cost = cost;
					min_bin_index = b;
					min_bin_dimension = dimension;

					aabb_left  = bounds_left [b];
					aabb_right = bounds_right[b];

					n_left  = count_left [b];
					n_right = count_right[b];

					min_bin_plane_distance = bounds_min + bounds_step * float(b);
				}
			}
		}

		split_dimension = min_bin_dimension;
		split_cost      = min_bin_cost;
		
		return min_bin_index;
	}
}
