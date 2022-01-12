#pragma once
#include <cASSERT>

#include <cuda.h>

#include "CUDACall.h"

#include "Core/String.h"

struct CUDAModule {
	CUmodule module;

	struct Global {
		CUdeviceptr ptr;

		template<typename T>
		inline void set_value(const T & value) const {
			CUDACALL(cuMemcpyHtoD(ptr, &value, sizeof(T)));
		}

		template<typename T>
		inline void set_value_async(const T & value, CUstream stream) const {
			CUDACALL(cuMemcpyHtoDAsync(ptr, &value, sizeof(T), stream));
		}

		template<typename T>
		inline T get_value() const {
			T result;
			CUDACALL(cuMemcpyDtoH(&result, ptr, sizeof(T)));

			return result;
		}
	};

	void init(const String & filename, int compute_capability, int max_registers);
	void free();

	Global get_global(const char * variable_name) const;
};
