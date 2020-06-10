#pragma once
#include "CUDACall.h"

namespace CUDAContext {
	inline int compute_capability = -1;

	inline unsigned long long total_memory;

	// Creates a new CUDA Context
	void init();

	// Available memory on GPU in bytes
	unsigned long long get_available_memory();

	// Creates a CUDA Array that is mapped to the given GL Texture handle
	CUarray map_gl_texture(unsigned gl_texture, unsigned flags);
}
