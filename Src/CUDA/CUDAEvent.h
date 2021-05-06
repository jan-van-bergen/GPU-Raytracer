#pragma once
#include <cuda.h>

#include "CUDACall.h"

struct CUDAEvent {
	inline static std::vector<CUDAEvent> event_pool;
	inline static int                    event_pool_num_used;
	
	CUevent event;

	struct Info {
		int display_order;
		const char * category;
		const char * name;
	} info;

	inline static void record(const Info & info, CUstream stream = nullptr) {
		if (event_pool_num_used == event_pool.size()) {
			CUDAEvent event = { };
			CUDACALL(cuEventCreate(&event.event, CU_EVENT_DEFAULT));
			event_pool.push_back(event);
		}

		CUDAEvent & event = event_pool[event_pool_num_used++];
		event.info = info;
		cuEventRecord(event.event, stream);
	}

	inline static void reset_pool() {
		event_pool_num_used = 0;
	}

	inline static float time_elapsed_between(const CUDAEvent & start, const CUDAEvent & end) {
		float result;
		CUDACALL(cuEventElapsedTime(&result, start.event, end.event));

		return result;
	}
};
