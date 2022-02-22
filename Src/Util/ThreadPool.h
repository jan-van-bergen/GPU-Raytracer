#pragma once
#include <thread>
#include <mutex>

#include "Core/Array.h"
#include "Core/Queue.h"
#include "Core/Function.h"

struct ThreadPool {
	using Work = Function<void()>;

private:
	Array<std::thread> threads;

	Queue<Work> work_queue;

	struct Signal {
		std::condition_variable condition;
		std::mutex              mutex;
	};

	Signal signal_submit;
	Signal signal_done;

	int              num_submitted = 0;
	std::atomic<int> num_done      = 0;

	std::atomic<bool> is_done;

public:
	ThreadPool(int thread_count = std::thread::hardware_concurrency());
	~ThreadPool();

	void submit(Work && work);

	void sync();
};
