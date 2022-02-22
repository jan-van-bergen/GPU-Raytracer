#pragma once
#include "Allocator.h"

// Linear burn-through Allocator that uses a small in-situ buffer that resides on the program stack
// A fallback Allocator can be provided when the StackAllocator runs out of space
template<size_t Size>
struct StackAllocator final : Allocator {
	Allocator * fallback_allocator = nullptr;

	size_t offset = 0;
	char data[Size];

	StackAllocator(Allocator * fallback_allocator = nullptr) : fallback_allocator(fallback_allocator) { }

	NON_COPYABLE(StackAllocator);
	NON_MOVEABLE(StackAllocator);

	~StackAllocator() = default;

private:
	char * alloc(size_t num_bytes) override {
		if (offset + num_bytes <= Size) {
			char * result = data + offset;
			offset += num_bytes;
			return result;
		} else {
			return Allocator::alloc_array<char>(fallback_allocator, num_bytes);
		}
	}

	void free(void * ptr) override {
		if (ptr >= data && ptr < data + Size) {
			// Do nothing
		} else {
			Allocator::free_array(fallback_allocator, ptr);
		}
	}
};
