#pragma once
#include <cassert>

#include <cuda.h>

#include "CUDACall.h"

#include "Texture.h"

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

		// Creates a new buffer on the Device, 
		// copies the given buffer over from the Host, 
		// and finally sets the value of this Global to the address of the buffer on the Device
		template<typename T>
		inline void set_buffer(const T * buffer, int count) const {
			CUDAMemory::Ptr<T> ptr = CUDAMemory::malloc<T>(count);
			CUDAMemory::memcpy(ptr, buffer, count);

			set_value(ptr);
		}

		template<typename T, int N>
		inline void set_buffer(const T (& buffer)[N]) const {
			set_buffer(buffer, N);
		}

		template<typename T>
		inline void set_buffer(const std::vector<T> & buffer) const {
			set_buffer(buffer.data(), buffer.size());
		}
	};

	void init(const char * filename, int compute_capability, int max_registers);

	void set_surface(const char * surface_name, CUarray array) const;

	void set_texture(const char * texture_name, CUarray array, CUfilter_mode filter, CUarray_format format, int channels) const;
	void set_texture(const char * texture_name, const Texture * texture) const;

	Global get_global(const char * variable_name) const;
};
