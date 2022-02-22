#pragma once
#include "Allocator.h"

#include "Device/CUDAMemory.h"

// Allocator that allocates CUDA pinned memory
// This is memory that is guarenteed not to be paged out by the OS
// This memory should be used to upload data to the GPU
// See: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
struct PinnedAllocator final : Allocator {
private:
	PinnedAllocator() { }

	char * alloc(size_t num_bytes) override {
		return CUDAMemory::malloc_pinned<char>(num_bytes);
	}

	void free(void * ptr) override {
		CUDAMemory::free_pinned(ptr);
	}

public:
	static PinnedAllocator * instance() {
		static PinnedAllocator allocator = { };
		return &allocator;
	}
};
