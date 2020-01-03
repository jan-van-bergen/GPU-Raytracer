#pragma once
#include "CUDACall.h"

namespace CUDAMemory {
	template<typename T>
	CUdeviceptr malloc(int count = 1) {
		assert(count > 0);

		CUdeviceptr ptr;
		CUDACALL(cuMemAlloc(&ptr, count * sizeof(T)));

		return ptr;
	}

	template<typename T>
	void memcpy(CUdeviceptr ptr, const T * data, int count = 1) {
		assert(data);
		assert(count > 0);

		CUDACALL(cuMemcpyHtoD(ptr, data, count * sizeof(T)));
	}
}
