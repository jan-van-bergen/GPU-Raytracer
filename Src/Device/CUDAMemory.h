#pragma once
#include "CUDACall.h"

#include "Core/Array.h"
#include "Core/Assertion.h"
#include "Core/IO.h"

namespace CUDAMemory {
	// Type safe device pointer wrapper
	template<typename T>
	struct Ptr {
		CUdeviceptr ptr;

		Ptr()                : ptr(NULL) { }
		Ptr(CUdeviceptr ptr) : ptr(ptr)  { }

		void operator=(Ptr other) {
			if (ptr != NULL) {
				IO::print("WARNING: CUDA memory leak detected!\n"_sv);
				__debugbreak();
			}
			ptr = other.ptr;
		}
	};

	template<typename T>
	inline T * malloc_pinned(size_t count = 1) {
		ASSERT(count > 0);

		T * ptr;
		CUDACALL(cuMemAllocHost(reinterpret_cast<void **>(&ptr), count * sizeof(T)));

		return ptr;
	}

	template<typename T>
	inline Ptr<T> malloc(size_t count = 1) {
		ASSERT(count > 0);

		CUdeviceptr ptr;
		CUDACALL(cuMemAlloc(&ptr, count * sizeof(T)));

		return Ptr<T>(ptr);
	}

	template<typename T>
	inline Ptr<T> malloc(const T * data, size_t count) {
		Ptr<T> ptr = malloc<T>(count);
		memcpy(ptr, data, count);

		return ptr;
	}

	template<typename T, int N>
	inline Ptr<T> malloc(const T (& data)[N]) {
		return malloc(data, N);
	}

	template<typename T>
	inline Ptr<T> malloc(const Array<T> & data) {
		return malloc(data.data(), data.size());
	}

	template<typename T>
	inline void free_pinned(T * ptr) {
		ASSERT(ptr);
		CUDACALL(cuMemFreeHost(ptr));
	}

	template<typename T>
	inline void free(Ptr<T> & ptr) {
		ASSERT(ptr.ptr);
		CUDACALL(cuMemFree(ptr.ptr));
		ptr.ptr = NULL;
	}

	template<typename T>
	inline void memcpy(Ptr<T> dst, const T * src, size_t count = 1) {
		ASSERT(src);
		ASSERT(dst.ptr);
		ASSERT(count > 0);

		CUDACALL(cuMemcpyHtoD(dst.ptr, src, count * sizeof(T)));
	}

	template<typename T>
	inline void memcpy_async(Ptr<T> dst, const T * src, size_t count, CUstream stream) {
		ASSERT(src);
		ASSERT(dst.ptr);
		ASSERT(count > 0);

		CUDACALL(cuMemcpyHtoDAsync(dst.ptr, src, count * sizeof(T), stream));
	}

	template<typename T>
	inline void memcpy(T * dst, Ptr<T> src, size_t count = 1) {
		ASSERT(src.ptr);
		ASSERT(dst);
		ASSERT(count > 0);

		CUDACALL(cuMemcpyDtoH(dst, src.ptr, count * sizeof(T)));
	}

	template<typename T>
	inline void memcpy_async(T * dst, Ptr<T> src, size_t count, CUstream stream) {
		ASSERT(src.ptr);
		ASSERT(dst);
		ASSERT(count > 0);

		CUDACALL(cuMemcpyDtoHAsync(dst, src.ptr, count * sizeof(T), stream));
	}

	template<typename T>
	inline void memset_async(Ptr<T> ptr, int value, size_t count, CUstream stream) {
		int size_in_bytes = count * sizeof(T);
		if ((size_in_bytes & 3) == 0) {
			CUDACALL(cuMemsetD32Async(ptr.ptr, value, size_in_bytes >> 2, stream));
		} else {
			CUDACALL(cuMemsetD8Async(ptr.ptr, value, size_in_bytes, stream));
		}
	}

	CUarray          create_array       (int width, int height, int channels, CUarray_format format);
	CUmipmappedArray create_array_mipmap(int width, int height, int channels, CUarray_format format, int level_count);

	void free_array(CUarray array);
	void free_array(CUmipmappedArray array);

	// Copies data from the Host Texture to the Device Array
	void copy_array(CUarray array, int width_in_bytes, int height, const void * data);

	CUtexObject  create_texture(CUarray array, CUfilter_mode filter);
	CUsurfObject create_surface(CUarray array);

	void free_texture(CUtexObject  texture);
	void free_surface(CUsurfObject surface);

	// Graphics Resource management (for OpenGL interop)
	CUgraphicsResource resource_register(unsigned gl_texture, unsigned flags);
	void               resource_unregister(CUgraphicsResource resource);

	CUarray resource_get_array(CUgraphicsResource resource);
}

template<typename T>
CUDAMemory::Ptr<T> operator+(CUDAMemory::Ptr<T> ptr, size_t offset) {
	return ptr.ptr + offset * sizeof(T);
}
