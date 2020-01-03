#pragma once
#include <cassert>

#include <cuda.h>

#include "CUDACall.h"

struct CUDAModule {
	CUmodule module;

	void init(const char * filename, int compute_capability);

	struct Global {
		const char * name;

		CUdeviceptr ptr;
		size_t size;

		template<typename T>
		inline void set(const T & value) const {
			assert(sizeof(T) <= size);

			CUDACALL(cuMemcpyHtoD(ptr, &value, size));
		}
	};

	inline Global get_global(const char * variable_name) const {
		Global global;
		global.name = variable_name;

		CUDACALL(cuModuleGetGlobal(&global.ptr, &global.size, module, global.name));

		return global;
	}
};
