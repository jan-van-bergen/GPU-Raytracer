#pragma once
#include "CUDACall.h"

namespace CUDAContext {
	inline int compute_capability = -1;

	inline unsigned long long total_memory;

	// Creates a new CUDA Context
	void init();
	void destroy();

	// Available memory on GPU in bytes
	unsigned long long get_available_memory();
}
