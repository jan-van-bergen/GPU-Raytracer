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
