#pragma once
#include <cuda.h>

#include "Util/IO.h"

#define CHECK_CUDA_CALLS true

#if CHECK_CUDA_CALLS
#define CUDACALL(result) check_cuda_call(result, __FILE__, __LINE__);

inline void check_cuda_call(CUresult result, const char * file, int line) {
	if (result != CUDA_SUCCESS) {
		const char * error_name;
		const char * error_string;

		cuGetErrorName  (result, &error_name);
		cuGetErrorString(result, &error_string);

		IO::print("{}:{}: CUDA call failed with error {}!\n{}\n"sv, file, line, error_name, error_string);
		__debugbreak();
	}
}
#else
#define CUDACALL(result) result
#endif
