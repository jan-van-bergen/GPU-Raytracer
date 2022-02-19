#pragma once
#include <new>

#include "Util/Util.h"

#define KILOBYTE(n) (n * 1024)
#define MEGABYTE(n) (n * 1024 * 1024)
#define GIGABYTE(n) (n * 1024 * 1024 * 1024)

struct Allocator {
	Allocator() = default;

	NON_COPYABLE(Allocator);
	NON_MOVEABLE(Allocator);

	virtual ~Allocator() { }

	template<typename T, typename ... Args>
	static T * alloc(Allocator * allocator, Args && ... args) {
		if (allocator) {
			return new (allocator->alloc(sizeof(T))) T { std::forward<Args>(args) ... };
		} else {
			return new T { std::forward<Args>(args) ... };
		}
	}

	template<typename T, typename ... Args>
	static T * alloc_array(Allocator * allocator, size_t count) {
		if (allocator) {
			return new (allocator->alloc(count * sizeof(T))) T[count];
		} else {
			return new T[count];
		}
	}

	template<typename T>
	static void free(Allocator * allocator, T * ptr) {
		if (allocator) {
			allocator->free(ptr);
		} else {
			delete ptr;
		}
	}

	template<typename T>
	static void free_array(Allocator * allocator, T * ptr) {
		if (allocator) {
			allocator->free(ptr);
		} else {
			delete [] ptr;
		}
	}

protected:
	virtual char * alloc(size_t num_bytes) = 0;
	virtual void   free (void * ptr)       = 0;
};
