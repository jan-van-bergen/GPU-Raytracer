#pragma once
#include "CUDACall.h"

namespace CUDAMemory {
	// Type safe device pointer wrapper
	template<typename T>
	struct Ptr {
		CUdeviceptr ptr;

		Ptr()                : ptr(NULL) { }
		Ptr(CUdeviceptr ptr) : ptr(ptr)  { }
	};

	template<typename T>
	inline Ptr<T> malloc(int count = 1) {
		assert(count > 0);

		CUdeviceptr ptr;
		CUDACALL(cuMemAlloc(&ptr, count * sizeof(T)));

		return Ptr<T>(ptr);
	}

	template<typename T>
	inline void memcpy(Ptr<T> ptr, const T * data, int count = 1) {
		assert(data);
		assert(count > 0);

		CUDACALL(cuMemcpyHtoD(ptr.ptr, data, count * sizeof(T)));
	}

	CUarray create_array  (int width, int height,            int channels, CUarray_format format);
	CUarray create_array3d(int width, int height, int depth, int channels, CUarray_format format, unsigned flags);

	// Copies data from the Host Texture to the Device Array
	void copy_array(CUarray array, int width_in_bytes, int height, const void * data);
}
