#pragma once
#include <cassert>

#include <cuda.h>

#include "CUDACall.h"

struct CUDAEvent {
	CUevent event;

	inline void init() {
		CUDACALL(cuEventCreate(&event, CU_EVENT_DEFAULT));
	}

	inline void record(CUstream stream = nullptr) {
		CUDACALL(cuEventRecord(event, stream));
	}

	inline static float time_elapsed_between(const CUDAEvent & start, const CUDAEvent & end) {
		float result;
		CUDACALL(cuEventElapsedTime(&result, start.event, end.event));

		return result;
	}
};
