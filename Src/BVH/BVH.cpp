#include "BVH.h"

#include "BVH/Builders/QBVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

OwnPtr<BVH> BVH::create_from_bvh2(BVH2 bvh) {
	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: {
			return OwnPtr<BVH2>::make(std::move(bvh));
		}
		case BVHType::QBVH: {
			OwnPtr<BVH4> bvh4 = OwnPtr<BVH4>::make();

			// Collapse binary BVH into quaternary BVH
			QBVHBuilder qbvh_builder = { };
			qbvh_builder.init(bvh4.get(), std::move(bvh));
			qbvh_builder.build(bvh);

			return bvh4;
		}
		case BVHType::CWBVH: {
			OwnPtr<BVH8> bvh8 = OwnPtr<BVH8>::make();

			// Collapse binary BVH into 8-way Compressed Wide BVH
			CWBVHBuilder cwbvh_builder = { };
			cwbvh_builder.init(bvh8.get(), std::move(bvh));
			cwbvh_builder.build(bvh);

			return bvh8;
		}
		default: abort();
	}
}
