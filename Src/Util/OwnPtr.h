#pragma once
#include "Assertion.h"

template<typename T>
struct OwnPtr {
	T * ptr;

	OwnPtr() : ptr(nullptr) { }

	explicit OwnPtr(T * ptr) : ptr(ptr) { }

	OwnPtr(const OwnPtr & other) = delete;

	OwnPtr(OwnPtr && other) {
		ptr = other.ptr;
		other.ptr = nullptr;
	}

	template<typename U>
	OwnPtr(OwnPtr<U> && other) {
		ptr = other.ptr;
		other.ptr = nullptr;
	}

	~OwnPtr() {
		release();
	}

	template<typename ... Args>
	static OwnPtr<T> make(Args ... args) {
		return OwnPtr(new T(std::forward<Args>(args) ...));
	}

	void release() {
		if (ptr) {
			ptr->~T();
			delete ptr;
			ptr = nullptr;
		}
	}

	      T * get()       { return ptr; }
	const T * get() const { return ptr; }

	OwnPtr & operator=(const OwnPtr & other) = delete;

	OwnPtr & operator=(OwnPtr && other) {
		release();
		ptr = other.ptr;
		other.ptr = nullptr;
		return *this;
	}

	T * operator->() {
		ASSERT(ptr);
		return ptr;
	}

	const T * operator->() const {
		ASSERT(ptr);
		return ptr;
	}

	operator bool() { return ptr != nullptr; }
};
