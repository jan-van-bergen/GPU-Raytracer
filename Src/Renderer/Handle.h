#pragma once
#include "CUDA/Common.h"

template<typename T>
struct Handle {
	int handle = INVALID;

	static Handle<T> get_default() { return Handle<T> { 0 }; }
};

template<typename T> bool operator==(Handle<T> a, Handle<T> b) { return a.handle == b.handle; }
template<typename T> bool operator!=(Handle<T> a, Handle<T> b) { return a.handle != b.handle; }
