#pragma once
#include <cuda.h>

struct CUDAModule {
	CUmodule module;

	void init(const char * filename, int compute_capability);
};
