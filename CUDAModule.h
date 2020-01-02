#pragma once
#include <cuda.h>

struct CUDAModule {
	CUmodule module;

	void init(const char * filename, int compute_capability);

	void bind_surface_to_texture(const char * surface_name, unsigned gl_texture) const;
};
