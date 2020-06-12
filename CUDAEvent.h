#pragma once
#include <cuda.h>

#include "CUDACall.h"

struct CUDAEvent {
	CUevent event;

	const char * category;
	const char * name;

	inline void init(const char * category, const char * name) {
		CUDACALL(cuEventCreate(&event, CU_EVENT_DEFAULT));

		this->category = category;
		this->name     = name;
	}

	inline void record(CUstream stream = nullptr) const {
		CUDACALL(cuEventRecord(event, stream));
	}

	inline static float time_elapsed_between(const CUDAEvent & start, const CUDAEvent & end) {
		float result;
		CUDACALL(cuEventElapsedTime(&result, start.event, end.event));

		return result;
	}
};
