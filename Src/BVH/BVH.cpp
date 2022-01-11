#include "BVH.h"

#include "BVH/Builders/QBVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

BVH * BVH::create_from_bvh2(BVH2 bvh) {
	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: {
			return new BVH2(std::move(bvh));
		}
		case BVHType::QBVH: {
			BVH4 * bvh4 = new BVH4();

			// Collapse binary BVH into quaternary BVH
			QBVHBuilder qbvh_builder = { };
			qbvh_builder.init(bvh4, std::move(bvh));
			qbvh_builder.build(bvh);

			return bvh4;
		}
		case BVHType::CWBVH: {
			BVH8 * bvh8 = new BVH8();

			// Collapse binary BVH into 8-way Compressed Wide BVH
			CWBVHBuilder cwbvh_builder = { };
			cwbvh_builder.init(bvh8, std::move(bvh));
			cwbvh_builder.build(bvh);

			return bvh8;
		}
		default: abort();
	}
}

void BVH2::aggregate(BVHNodePtr aggregated_bvh_nodes, int index_offset, int bvh_offset) const {
	BVHNode2 * dst = aggregated_bvh_nodes._2 + bvh_offset;

	for (size_t n = 0; n < nodes.size(); n++) {
		BVHNode2 & node = dst[n];
		node = nodes[n];

		if (node.is_leaf()) {
			node.first += index_offset;
		} else {
			node.left += bvh_offset;
		}
	}
}

void BVH4::aggregate(BVHNodePtr aggregated_bvh_nodes, int index_offset, int bvh_offset) const {
	BVHNode4 * dst = aggregated_bvh_nodes._4 + bvh_offset;

	for (int n = 0; n < nodes.size(); n++) {
		BVHNode4 & node = dst[n];
		node = nodes[n];

		int child_count = node.get_child_count();
		for (int c = 0; c < child_count; c++) {
			if (node.is_leaf(c)) {
				node.get_index(c) += index_offset;
			} else {
				node.get_index(c) += bvh_offset;
			}
		}
	}
}

void BVH8::aggregate(BVHNodePtr aggregated_bvh_nodes, int index_offset, int bvh_offset) const {
	BVHNode8 * dst = aggregated_bvh_nodes._8 + bvh_offset;

	for (int n = 0; n < nodes.size(); n++) {
		BVHNode8 & node = dst[n];
		node = nodes[n];

		node.base_index_triangle += index_offset;
		node.base_index_child    += bvh_offset;
	}
}
