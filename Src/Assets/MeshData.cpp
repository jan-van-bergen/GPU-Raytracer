#include "MeshData.h"

#include "Assets/OBJLoader.h"

#include "BVH/Builders/QBVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

void MeshData::init_bvh(const BVH & bvh) {
#if BVH_TYPE == BVH_BVH || BVH_TYPE == BVH_SBVH
	// No collapse needed
	this->bvh = bvh;
#elif BVH_TYPE == BVH_QBVH
	// Collapse binary BVH into quaternary BVH
	QBVHBuilder qbvh_builder;
	qbvh_builder.init(&this->bvh, bvh);
	qbvh_builder.build(bvh);
	
	delete [] bvh.nodes;
#elif BVH_TYPE == BVH_CWBVH
	// Collapse binary BVH into 8-way Compressed Wide BVH
	CWBVHBuilder cwbvh_builder;
	cwbvh_builder.init(&this->bvh, bvh);
	cwbvh_builder.build(bvh);
	cwbvh_builder.free();

	delete [] bvh.indices;
	delete [] bvh.nodes;
#endif
}
