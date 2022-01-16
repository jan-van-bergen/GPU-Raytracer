#pragma once
#include "Assertion.h"

template<typename T>
struct OwnPtr {
	T * ptr;

	OwnPtr() : ptr(nullptr) { }

	explicit OwnPtr(T * ptr) : ptr(ptr) { }

	OwnPtr(const OwnPtr & other) = delete;

	OwnPtr(OwnPtr && other) noexcept {
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

	void release() {
		if (ptr) {
			delete ptr;
			ptr = nullptr;
		}
	}

	      T * get()       { return ptr; }
	const T * get() const { return ptr; }

	OwnPtr & operator=(const OwnPtr & other) = delete;

	OwnPtr & operator=(OwnPtr && other) noexcept {
		release();
		ptr = other.ptr;
		other.ptr = nullptr;
		return *this;
	}

	template<typename S>
	OwnPtr & operator=(OwnPtr<S> && other) {
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

template<typename T, typename ... Args>
inline OwnPtr<T> make_owned(Args && ... args) {
	return OwnPtr<T>(new T { std::forward<Args>(args) ... });
}
