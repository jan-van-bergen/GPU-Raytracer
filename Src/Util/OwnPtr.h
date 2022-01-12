#pragma once
#include "Assertion.h"

template<typename T>
struct OwnPtr {
	T * ptr;

	OwnPtr() : ptr(nullptr) { }

	OwnPtr(T * ptr) : ptr(ptr) { }

	OwnPtr(const OwnPtr & other) = delete;

	OwnPtr(OwnPtr && other) {
		release();
		ptr = other.ptr;
		other.ptr = nullptr;
	}

	~OwnPtr() {
		release();
	}

	void release() {
		if (ptr) {
			ptr->~T();
			delete ptr;
			ptr = nullptr;
		}
	}

	OwnPtr & operator=(const OwnPtr & other) = delete;

	OwnPtr & operator=(OwnPtr && other) {
		release();
		ptr = other.ptr;
		other.ptr = nullptr;
	}

	T * operator->() {
		ASSERT(ptr);
		return ptr;
	}

	const T * operator->() const {
		ASSERT(ptr);
		return ptr;
	}

	operator const T*() const { return ptr; }
    operator       T*()       { return ptr; }

	operator bool() { return ptr != nullptr; }
};
