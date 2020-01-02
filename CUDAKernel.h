#pragma once
#include <cstring>

#include <cuda.h>

#include "CUDAModule.h"

struct CUDAKernel {
	const CUDAModule * module;
	CUfunction kernel;

	unsigned char * parameter_buffer;
	
	int  grid_dim_x = 64,  grid_dim_y = 1,  grid_dim_z = 1;
	int block_dim_x = 64, block_dim_y = 1, block_dim_z = 1;
	
	void init(const CUDAModule * module, const char * kernel_name) {
		this->module = module;

		CUDACALL(cuModuleGetFunction(&kernel, module->module, kernel_name));

		CUDACALL(cuFuncSetCacheConfig    (kernel, CU_FUNC_CACHE_PREFER_L1));
		CUDACALL(cuFuncSetSharedMemConfig(kernel, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));

		parameter_buffer = new unsigned char[32 * 64];
	}

	// Execute kernel without parameters
	void execute() {
		execute_internal(0);
	}

	// Exectute kernel with one ore more parameters
	template<typename ... T>
	void execute(const T & ... parameters) {
		int buffer_size = fill_buffer(0, parameters...);
		
		execute_internal(buffer_size);
	}

	void set_surface(const char * surface_name, CUarray array) const {
		CUsurfref surface;
		CUDACALL(cuModuleGetSurfRef(&surface, module->module, surface_name));

		CUDACALL(cuSurfRefSetArray(surface, reinterpret_cast<CUarray>(array), 0));
	}

	void set_grid_dim(int x, int y, int z) {
		grid_dim_x = x;
		grid_dim_y = y;
		grid_dim_z = z;
	}

	void set_block_dim(int x, int y, int z) {
		block_dim_x = x;
		block_dim_y = y;
		block_dim_z = z;
	}

private:
	template<typename T>
	int fill_buffer(int buffer_offset, const T & parameter) {
		int size  = sizeof(T);
		int align = alignof(T);

		int alignment = buffer_offset & (align - 1);
		if (alignment != 0) {
			buffer_offset += align - alignment;
		}

		memcpy(parameter_buffer + buffer_offset, &parameter, size);

		return buffer_offset + size;
	}

	template<typename T, typename ... Ts>
	int fill_buffer(int buffer_offset, const T & parameter, const Ts & ... parameters) {
		int offset = fill_buffer(buffer_offset, parameter);

		return fill_buffer(offset, parameters...);
	}

	void execute_internal(int parameter_buffer_size) const {
		void * params[] = { 
			CU_LAUNCH_PARAM_BUFFER_POINTER, parameter_buffer, 
			CU_LAUNCH_PARAM_BUFFER_SIZE,   &parameter_buffer_size, 
			CU_LAUNCH_PARAM_END 
		};
		
		CUDACALL(cuLaunchKernel(kernel, 
			grid_dim_x, grid_dim_y, grid_dim_z, 
			block_dim_x, block_dim_y, block_dim_z, 
			0, NULL, NULL, params)
		);
	}
};
