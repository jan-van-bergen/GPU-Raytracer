#pragma once
#include <cuda.h>

#include "CUDACall.h"

struct CUDAEvent {
	CUevent event;

	struct Desc {
		int display_order;
		String category;
		String name;
	};
	const Desc * desc;

	CUDAEvent(CUevent event, const Desc * desc) : event(event), desc(desc) { }

	DEFAULT_COPYABLE(CUDAEvent);
	DEFAULT_MOVEABLE(CUDAEvent);

	~CUDAEvent() = default;

	inline static float time_elapsed_between(const CUDAEvent & start, const CUDAEvent & end) {
		float result;
		CUDACALL(cuEventElapsedTime(&result, start.event, end.event));

		return result;
	}
};

struct CUDAEventPool {
	Array<CUDAEvent> pool;
	size_t           num_used;

	void record(const CUDAEvent::Desc * event_desc, CUstream stream = nullptr) {
		// Check if the Pool is already using its maximum capacity
		if (num_used == pool.size()) {
			// Create new Event
			CUevent event = { };
			CUDACALL(cuEventCreate(&event, CU_EVENT_DEFAULT));
			pool.emplace_back(event, event_desc);
		}

		CUDAEvent & event = pool[num_used++];
		event.desc = event_desc;

		CUDACALL(cuEventRecord(event.event, stream));
	}

	void reset() {
		num_used = 0;
	}
};
