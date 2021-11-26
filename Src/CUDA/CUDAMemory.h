#pragma once
#include "CUDACall.h"

#include "Util/Array.h"

namespace CUDAMemory {
	// Type safe device pointer wrapper
	template<typename T>
	struct Ptr {
		CUdeviceptr ptr;

		Ptr()                : ptr(NULL) { }
		Ptr(CUdeviceptr ptr) : ptr(ptr)  { }

		void operator=(Ptr other) {
			if (ptr != NULL) {
				puts("WARNING: CUDA memory leak detected!");
				__debugbreak();
			}
			ptr = other.ptr;
		}
	};

	template<typename T>
	inline T * malloc_pinned(int count = 1) {
		assert(count > 0);

		T * ptr;
		CUDACALL(cuMemAllocHost(reinterpret_cast<void **>(&ptr), count * sizeof(T)));

		return ptr;
	}

	template<typename T>
	inline Ptr<T> malloc(int count = 1) {
		assert(count > 0);

		CUdeviceptr ptr;
		CUDACALL(cuMemAlloc(&ptr, count * sizeof(T)));

		return Ptr<T>(ptr);
	}

	template<typename T>
	inline Ptr<T> malloc(const T * data, int count) {
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
		assert(ptr);
		CUDACALL(cuMemFreeHost(ptr));
	}

	template<typename T>
	inline void free(Ptr<T> & ptr) {
		assert(ptr.ptr);
		CUDACALL(cuMemFree(ptr.ptr));
		ptr.ptr = NULL;
	}

	template<typename T>
	inline void memcpy(Ptr<T> ptr, const T * data, int count = 1) {
		assert(ptr.ptr);
		assert(data);
		assert(count > 0);

		CUDACALL(cuMemcpyHtoD(ptr.ptr, data, count * sizeof(T)));
	}

	template<typename T>
	inline void memcpy(T * data, Ptr<T> ptr, int count = 1) {
		assert(ptr.ptr);
		assert(data);
		assert(count > 0);

		CUDACALL(cuMemcpyDtoH(data, ptr.ptr, count * sizeof(T)));
	}

	template<typename T>
	inline void memset_async(Ptr<T> ptr, int value, int count, CUstream stream) {
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
