#pragma once
#include <cassert>

#include <cuda.h>

#include "CUDACall.h"

#include "Texture.h"

struct CUDAModule {
	CUmodule module;
	
	struct Global {
		const char * name;

		CUdeviceptr ptr;
		
		template<typename T>
		inline void set_value(const T & value) const {
			CUDACALL(cuMemcpyHtoD(ptr, &value, sizeof(T)));
		}

		template<typename T>
		inline T get_value() const {
			T result;
			CUDACALL(cuMemcpyDtoH(&result, ptr, sizeof(T)));

			return result;
		}
	};

	void init(const char * filename, int compute_capability);

	void set_surface(const char * surface_name, CUarray array) const;
	void set_texture(const char * texture_name, const Texture * texture) const;

	Global get_global(const char * variable_name) const;
};
