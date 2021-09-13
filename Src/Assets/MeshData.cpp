#include "MeshData.h"

#include "Config.h"

#include "Assets/OBJLoader.h"

#include "BVH/Builders/QBVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

void MeshData::init_bvh(BVH & bvh) {
	switch (config.bvh_type) {
		case BVHType::BVH:
		case BVHType::SBVH: {
			this->bvh = bvh;
			break;
		}
		case BVHType::QBVH: {
			// Collapse binary BVH into quaternary BVH
			QBVHBuilder qbvh_builder = { };
			qbvh_builder.init(&this->bvh, bvh);
			qbvh_builder.build(bvh);

			delete [] bvh.nodes_2; // Delete only the nodes array, indices array is cannibalized by the QBVH
			break;
		}
		case BVHType::CWBVH: {
			// Collapse binary BVH into 8-way Compressed Wide BVH
			CWBVHBuilder cwbvh_builder = { };
			cwbvh_builder.init(&this->bvh, bvh);
			cwbvh_builder.build(bvh);
			cwbvh_builder.free();

			delete [] bvh.indices;
			delete [] bvh.nodes_2;
			break;
		}
		default: abort();
	}
}
