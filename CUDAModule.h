#pragma once
#include <cassert>

#include <cuda.h>

#include "CUDACall.h"

struct CUDAModule {
	CUmodule module;

	void init(const char * filename, int compute_capability);

	template<typename T>
	inline void set_global(const char * variable_name, const T & value) const {
		CUdeviceptr ptr;
		size_t size;
		CUDACALL(cuModuleGetGlobal(&ptr, &size, module, variable_name));

		assert(sizeof(T) <= size);

		CUDACALL(cuMemcpyHtoD(ptr, &value, size));
	}
};
