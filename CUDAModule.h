#pragma once
#include <cstdio>

//#include <cuda_gl_interop.h>
#include <cudaGL.h>

#include "CUDACall.h"

struct CUDAModule {
	CUmodule module;

	inline void init(const char * filename, int compute_capability) {
		const int max_reg_count = 63;

		char output_filename[256];
		sprintf_s(output_filename, "%s.cubin", filename);

		char command[256];
#ifdef _DEBUG
		sprintf_s(command, "nvcc -cubin -use_fast_math -I=\"lib\\CUDA\" -Xptxas=\"-v\" -lineinfo -maxrregcount=%i -arch=sm_%i -o %s %s 2>>%s 1>>%s", max_reg_count, compute_capability, output_filename, filename, "errorlog.txt", "buildlog.txt");
#else
		sprintf_s(command, "nvcc -cubin -use_fast_math -I=\"lib\\CUDA\" -restrict -Xptxas=\"-v\" -maxrregcount=%i -arch=sm_%i -o %s %s 2>>%s 1>>%s", max_reg_count, compute_capability, output_filename, filename, "errorlog.txt", "buildlog.txt" );
#endif
		printf("Compiling CUDA Module %s\n", filename);

		int exit_code = system(command);
		if (exit_code != EXIT_SUCCESS) abort(); // Compilation failed!

		printf("Compilation finished!\n");

		CUDACALL(cuModuleLoad(&module, output_filename));
	}

	inline void bind_surface_to_texture(const char * surface_name, unsigned gl_texture) const {
		CUsurfref surface;
		CUgraphicsResource resource;
		CUDACALL(cuModuleGetSurfRef(&surface, module, surface_name));
		CUDACALL(cuGraphicsGLRegisterImage(&resource, gl_texture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));
	}
};
