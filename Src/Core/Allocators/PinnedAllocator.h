#pragma once
#include "Allocator.h"

#include "Device/CUDAMemory.h"

struct PinnedAllocator final : Allocator {
	char * alloc(size_t num_bytes) override {
		return CUDAMemory::malloc_pinned<char>(num_bytes);
	}

	void free(void * ptr) override {
		CUDAMemory::free_pinned(ptr);
	}

	static PinnedAllocator * instance() {
		static PinnedAllocator allocator = { };
		return &allocator;
	}
};
