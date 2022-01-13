#include "BVH.h"

#include "Core/IO.h"
#include "Core/ScopeTimer.h"

#include "BVH/Builders/SAHBuilder.h"
#include "BVH/Builders/SBVHBuilder.h"
#include "BVH/Converters/BVH4Converter.h"
#include "BVH/Converters/BVH8Converter.h"

#include "BVH/BVHOptimizer.h"

BVH2 BVH::create_from_triangles(const Array<Triangle> & triangles) {
	IO::print("Constructing BVH...\r"_sv);

	BVH2 bvh = { };

	// Only the SBVH uses SBVH as its starting point,
	// all other BVH types use the standard BVH as their starting point
	if (config.bvh_type == BVHType::SBVH) {
		ScopeTimer timer("SBVH Construction"_sv);

		SBVHBuilder(bvh, triangles.size()).build(triangles);
	} else  {
		ScopeTimer timer("BVH Construction"_sv);

		SAHBuilder(bvh, triangles.size()).build(triangles);
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
			return make_owned<BVH2>(std::move(bvh));
		}
		case BVHType::BVH4: {
			// Collapse binary BVH into 4-way BVH
			OwnPtr<BVH4> bvh4 = make_owned<BVH4>();
			BVH4Converter(*bvh4.get(), bvh).convert();
			return bvh4;
		}
		case BVHType::BVH8: {
			// Collapse binary BVH into 8-way Compressed Wide BVH
			OwnPtr<BVH8> bvh8 = make_owned<BVH8>();
			BVH8Converter(*bvh8.get(), bvh).convert();
			return bvh8;
		}
		default: abort();
	}
}
