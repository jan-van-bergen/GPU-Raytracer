#include "BVH.h"

void BVH::aggregate(char * aggregated_bvh_nodes, int index_offset, int bvh_offset) const {
	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: {
			BVHNode2 * dst = reinterpret_cast<BVHNode2 *>(aggregated_bvh_nodes) + bvh_offset;

			for (int n = 0; n < node_count; n++) {
				BVHNode2 & node = dst[n];
				node = nodes_2[n];

				if (node.is_leaf()) {
					node.first += index_offset;
				} else {
					node.left += bvh_offset;
				}
			}
			break;
		}
		case BVHType::QBVH: {
			BVHNode4 * dst = reinterpret_cast<BVHNode4 *>(aggregated_bvh_nodes) + bvh_offset;

			for (int n = 0; n < node_count; n++) {
				BVHNode4 & node = dst[n];
				node = nodes_4[n];

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
			BVHNode8 * dst = reinterpret_cast<BVHNode8 *>(aggregated_bvh_nodes) + bvh_offset;

			for (int n = 0; n < node_count; n++) {
				BVHNode8 & node = dst[n];
				node = nodes_8[n];

				node.base_index_triangle += index_offset;
				node.base_index_child    += bvh_offset;
			}
			break;
		}
	}
}
