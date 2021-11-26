#include "BVH.h"

void BVH::aggregate(BVHNodePtr aggregated_bvh_nodes, int index_offset, int bvh_offset) const {
	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: {
			BVHNode2 * dst = aggregated_bvh_nodes._2 + bvh_offset;

			for (int n = 0; n < node_count; n++) {
				BVHNode2 & node = dst[n];
				node = nodes._2[n];

				if (node.is_leaf()) {
					node.first += index_offset;
				} else {
					node.left += bvh_offset;
				}
			}
			break;
		}
		case BVHType::QBVH: {
			BVHNode4 * dst = aggregated_bvh_nodes._4 + bvh_offset;

			for (int n = 0; n < node_count; n++) {
				BVHNode4 & node = dst[n];
				node = nodes._4[n];

				int child_count = node.get_child_count();
				for (int c = 0; c < child_count; c++) {
					if (node.is_leaf(c)) {
						node.get_index(c) += index_offset;
					} else {
						node.get_index(c) += bvh_offset;
					}
				}
			}
			break;
		}
		case BVHType::CWBVH: {
			BVHNode8 * dst = aggregated_bvh_nodes._8 + bvh_offset;

			for (int n = 0; n < node_count; n++) {
				BVHNode8 & node = dst[n];
				node = nodes._8[n];

				node.base_index_triangle += index_offset;
				node.base_index_child    += bvh_offset;
			}
			break;
		}
	}
}
