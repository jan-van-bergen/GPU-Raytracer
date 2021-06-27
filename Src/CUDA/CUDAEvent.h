#pragma once
#include <cuda.h>

#include "CUDACall.h"

struct CUDAEvent {
	CUevent event;

	struct Desc {
		int display_order;
		const char * category;
		const char * name;
	} desc;

	inline static float time_elapsed_between(const CUDAEvent & start, const CUDAEvent & end) {
		float result;
		CUDACALL(cuEventElapsedTime(&result, start.event, end.event));

		return result;
	}
};

struct CUDAEventPool {
	std::vector<CUDAEvent> pool;
	int                    num_used;

	inline void record(const CUDAEvent::Desc & event_desc, CUstream stream = nullptr) {
		// Check if the Pool is already using its maximum capacity
		if (num_used == pool.size()) {
			// Create new Event
			CUDAEvent event = { };
			CUDACALL(cuEventCreate(&event.event, CU_EVENT_DEFAULT));
			pool.push_back(event);
		}

		CUDAEvent & event = pool[num_used++];
		event.desc = event_desc;
		CUDACALL(cuEventRecord(event.event, stream));
	}

	inline void reset() {
		num_used = 0;
	}
};
