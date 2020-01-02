#pragma once
#include <cstdio>

#include <cuda.h>

#define CHECK_CUDA_CALLS true

#if CHECK_CUDA_CALLS
#define CUDACALL(result) check_cuda_call(result, __FILE__, __LINE__);

inline void check_cuda_call(CUresult result, const char * file, int line) {
	if (result != CUDA_SUCCESS) {
		const char * error_name;
		const char * error_string;

		cuGetErrorName  (result, &error_name);
		cuGetErrorString(result, &error_string);

		printf("CUDA call at %s line %i failed with error %s!\n%s", file, line, error_name, error_string);
      
		__debugbreak();
	}
}
#else
#define CUDACALL(result) result
#endif
