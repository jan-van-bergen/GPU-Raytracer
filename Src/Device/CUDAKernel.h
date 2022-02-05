#pragma once
#include <math.h>

#include <cuda.h>

#include "Core/IO.h"

#include "CUDAModule.h"

struct CUDAKernel {
	static constexpr int PARAMETER_BUFFER_SIZE = 256; // In bytes

	CUfunction kernel;

	mutable unsigned char parameter_buffer[PARAMETER_BUFFER_SIZE];

	int  grid_dim_x = 64,  grid_dim_y = 1,  grid_dim_z = 1;
	int block_dim_x = 64, block_dim_y = 1, block_dim_z = 1;

	unsigned shared_memory_bytes = 0;

	inline void init(const CUDAModule * module, const char * kernel_name) {
		CUresult result = cuModuleGetFunction(&kernel, module->module, kernel_name);
		if (result == CUDA_ERROR_NOT_FOUND) {
			IO::print("No Kernel with name '{}' was found in the Module!\n"_sv, kernel_name);
			IO::exit(1);
		}
		CUDACALL(result);

		CUDACALL(cuFuncSetCacheConfig    (kernel, CU_FUNC_CACHE_PREFER_L1));
		CUDACALL(cuFuncSetSharedMemConfig(kernel, CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE));
	}

	// Execute kernel without parameters
	inline void execute() {
		execute_internal(0);
	}

	// Exectute kernel with one ore more parameters
	template<typename ... T>
	inline void execute(const T & ... parameters) {
		int buffer_size = fill_buffer(0, parameters...);
		ASSERT(buffer_size < PARAMETER_BUFFER_SIZE);

		execute_internal(buffer_size);
	}

	inline void set_grid_dim(int x, int y, int z) {
		grid_dim_x = x;
		grid_dim_y = y;
		grid_dim_z = z;
	}

	inline void set_block_dim(int x, int y, int z) {
		block_dim_x = x;
		block_dim_y = y;
		block_dim_z = z;
	}

	inline void occupancy_max_block_size_1d() {
		int grid, block;
		CUDACALL(cuOccupancyMaxPotentialBlockSize(&grid, &block, kernel, nullptr, 0, 0));

		set_block_dim(block, 1, 1);
	}

	inline void occupancy_max_block_size_2d() {
		int grid, block;
		CUDACALL(cuOccupancyMaxPotentialBlockSize(&grid, &block, kernel, nullptr, 0, 0));

		// Take sqrt because we want block_x x block_y to be as square as possible
		int block_x = int(sqrt(block));
		// Make sure block_x is a multiple of 32
		block_x += (32 - block_x) & 31;

		if (block_x == 0) block_x = 32;

		int block_y = block / block_x;

		set_block_dim(block_x, block_y, 1);
	}

	inline void set_shared_memory(unsigned bytes) {
		shared_memory_bytes = bytes;
	}

private:
	template<typename T>
	inline int fill_buffer(int buffer_offset, const T & parameter) {
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
	inline int fill_buffer(int buffer_offset, const T & parameter, const Ts & ... parameters) {
		int offset = fill_buffer(buffer_offset, parameter);

		return fill_buffer(offset, parameters...);
	}

	inline void execute_internal(size_t parameter_buffer_size) const {
		void * params[] = {
			CU_LAUNCH_PARAM_BUFFER_POINTER, parameter_buffer,
			CU_LAUNCH_PARAM_BUFFER_SIZE,   &parameter_buffer_size,
			CU_LAUNCH_PARAM_END
		};

		CUDACALL(cuLaunchKernel(kernel,
			grid_dim_x,  grid_dim_y,  grid_dim_z,
			block_dim_x, block_dim_y, block_dim_z,
			shared_memory_bytes, nullptr, nullptr, params
		));
	}
};
