#pragma once
#include <cassert>

#include <cuda.h>

#include "CUDACall.h"

struct CUDAModule {
	CUmodule module;
	
	struct Global {
		const char * name;

		CUdeviceptr ptr;
		
		template<typename T>
		inline void set(const T & value) const {
			CUDACALL(cuMemcpyHtoD(ptr, &value, sizeof(T)));
		}
	};

	void init(const char * filename, int compute_capability);

	Global get_global(const char * variable_name) const;
};
