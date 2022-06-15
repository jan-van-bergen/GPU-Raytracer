#pragma once
#include "Allocator.h"

#include "Math/Math.h"

template<size_t Alignment>
struct AlignedAllocator final : Allocator {
	static_assert(Math::is_power_of_two(Alignment));

	static AlignedAllocator * instance() {
		static AlignedAllocator allocator = { };
		return &allocator;
	}

private:
	AlignedAllocator() = default;

	NON_COPYABLE(AlignedAllocator);
	NON_MOVEABLE(AlignedAllocator);

	~AlignedAllocator() = default;

	char * alloc(size_t num_bytes) override {
		return static_cast<char *>(_aligned_malloc(num_bytes, Alignment));
	}

	void free(void * ptr) override {
		_aligned_free(ptr);
	}
};
