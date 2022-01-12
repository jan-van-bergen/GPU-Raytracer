#include "BVH.h"

#include "Core/IO.h"
#include "Core/ScopeTimer.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/SBVHBuilder.h"
#include "BVH/Builders/QBVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

#include "BVH/BVHOptimizer.h"

BVH2 BVH::create_from_triangles(const Array<Triangle> & triangles) {
	IO::print("Constructing BVH...\r"sv);

	BVH2 bvh = { };

	// Only the SBVH uses SBVH as its starting point,
	// all other BVH types use the standard BVH as their starting point
	if (config.bvh_type == BVHType::SBVH) {
		ScopeTimer timer("SBVH Construction");

		SBVHBuilder(&bvh, triangles.size()).build(triangles);
	} else  {
		ScopeTimer timer("BVH Construction");

		BVHBuilder(&bvh, triangles.size()).build(triangles);
	}

	if (config.enable_bvh_optimization) {
		BVHOptimizer::optimize(bvh);
	}

	return bvh;
}

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
