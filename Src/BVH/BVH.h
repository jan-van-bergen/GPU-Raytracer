#pragma once
#include "Config.h"

#include "Renderer/Triangle.h"

#include "Core/Array.h"
#include "Core/OwnPtr.h"

typedef unsigned char byte;

struct BVHNode2 {
	AABB aabb;
	union {
		int left;
		int first;
	};
	unsigned count : 30;
	unsigned axis  : 2;

	inline bool is_leaf() const {
		return count > 0;
	}
};

struct BVHNode4 {
	float aabb_min_x[4] = { 0.0f };
	float aabb_min_y[4] = { 0.0f };
	float aabb_min_z[4] = { 0.0f };
	float aabb_max_x[4] = { 0.0f };
	float aabb_max_y[4] = { 0.0f };
	float aabb_max_z[4] = { 0.0f };

	struct {
		int index;
		int count;
	} index_and_count[4];

	inline       int & get_index(int i)       { return index_and_count[i].index; }
	inline const int & get_index(int i) const { return index_and_count[i].index; }
	inline       int & get_count(int i)       { return index_and_count[i].count; }
	inline const int & get_count(int i) const { return index_and_count[i].count; }

	inline bool is_leaf(int i) { return get_count(i) > 0; }

	inline int get_child_count() const {
		int result = 0;

		for (int i = 0; i < 4; i++) {
			if (get_count(i) == -1) break;

			result++;
		}

		return result;
	}

};

static_assert(sizeof(BVHNode4) == 128);

struct BVHNode8 {
	Vector3 p;
	byte e[3];
	byte imask;

	unsigned base_index_child;
	unsigned base_index_triangle;

	byte meta[8] = { };

	byte quantized_min_x[8] = { }, quantized_max_x[8] = { };
	byte quantized_min_y[8] = { }, quantized_max_y[8] = { };
	byte quantized_min_z[8] = { }, quantized_max_z[8] = { };

	inline bool is_leaf(int child_index) {
		return (meta[child_index] & 0b00011111) < 24;
	}
};

static_assert(sizeof(BVHNode8) == 80);

struct BVH2;

struct BVH {
	Array<int> indices;

	virtual ~BVH() = default;

	virtual size_t node_count() const = 0;

	static BVH2 create_from_triangles(const Array<Triangle> & triangles);

	static OwnPtr<BVH> create_from_bvh2(BVH2 bvh);

	static BVHType underlying_bvh_type() {
		// All BVH use standard BVH as underlying type, only SBVH uses SBVH
		if (cpu_config.bvh_type == BVHType::SBVH) {
			return BVHType::SBVH;
		} else {
			return BVHType::BVH;
		}
	}
};

struct BVH2 final : BVH {
	Array<BVHNode2> nodes;

	BVH2(Allocator * allocator = nullptr) : nodes(allocator) { }

	DEFAULT_COPYABLE(BVH2);
	DEFAULT_MOVEABLE(BVH2);

	size_t node_count() const override { return nodes.size(); }
};

struct BVH4 final : BVH {
	Array<BVHNode4> nodes;

	BVH4(Allocator * allocator = nullptr) : nodes(allocator) { }

	NON_COPYABLE(BVH4);
	DEFAULT_MOVEABLE(BVH4);

	size_t node_count() const override{ return nodes.size(); }
};

struct BVH8 final : BVH {
	Array<BVHNode8> nodes;

	BVH8(Allocator * allocator = nullptr) : nodes(allocator) { }

	size_t node_count() const override { return nodes.size(); }
};
