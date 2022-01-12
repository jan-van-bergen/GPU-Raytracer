#pragma once
#include <thread>
#include <mutex>
#include <functional>

#include "Core/Queue.h"

struct ThreadPool {
	using Work = std::function<void()>;

private:
	std::thread * threads;
	int           thread_count;

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
	void init(int thread_count = std::thread::hardware_concurrency());
	void free();

	void submit(Work && work);

	void sync();
};
