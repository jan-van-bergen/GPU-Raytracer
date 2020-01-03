#pragma once
#include "CUDACall.h"

namespace CUDAMemory {
	template<typename T>
	inline CUdeviceptr malloc(int count = 1) {
		assert(count > 0);

		CUdeviceptr ptr;
		CUDACALL(cuMemAlloc(&ptr, count * sizeof(T)));

		return ptr;
	}

	inline CUarray create_array(int width, int height, int channels, CUarray_format format) {
		CUDA_ARRAY_DESCRIPTOR desc;
		desc.Width  = width;
		desc.Height = height;
		desc.NumChannels = channels;
		desc.Format = format;

		CUarray array;
		CUDACALL(cuArrayCreate(&array, &desc));

		return array;
	}

	template<typename T>
	inline void memcpy(CUdeviceptr ptr, const T * data, int count = 1) {
		assert(data);
		assert(count > 0);

		CUDACALL(cuMemcpyHtoD(ptr, data, count * sizeof(T)));
	}
}
