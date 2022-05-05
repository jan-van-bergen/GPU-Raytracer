#include "ThreadPool.h"

#include <thread>
#include <mutex>

#include "Core/Array.h"
#include "Core/Queue.h"

static Array<std::thread> threads;

static Queue<ThreadPool::Work> work_queue;

struct Signal {
	std::condition_variable condition;
	std::mutex              mutex;
};

static Signal signal_submit;
static Signal signal_done;

static int              num_submitted = 0;
static std::atomic<int> num_done      = 0;

static std::atomic<bool> is_done;

void ThreadPool::init() {
	init(std::thread::hardware_concurrency());
}

void ThreadPool::init(int thread_count) {
	threads.resize(thread_count);

	for (size_t i = 0; i < threads.size(); i++) {
		threads[i] = std::thread([]() {
			while (true) {
				Work work;
				{
					std::unique_lock<std::mutex> lock(signal_submit.mutex);
					signal_submit.condition.wait(lock, []{ return !work_queue.is_empty() || is_done; });

					if (is_done) return;

					work = work_queue.pop();
				}
				work();

				num_done++;
				signal_done.condition.notify_one();
			}
		});
	}

	num_submitted = 0;
	num_done      = 0;
}

void ThreadPool::free() {
	{
		std::lock_guard<std::mutex> lock(signal_submit.mutex);
		is_done = true;
	}
	signal_submit.condition.notify_all();

	for (size_t i = 0; i < threads.size(); i++) {
		threads[i].join();
	}
}

void ThreadPool::submit(Work && work) {
	num_submitted++;

	{
		std::lock_guard<std::mutex> lock(signal_submit.mutex);
		work_queue.push(std::move(work));
	}
	signal_submit.condition.notify_one();
}

void ThreadPool::sync() {
	std::unique_lock<std::mutex> lock(signal_done.mutex);
	signal_done.condition.wait(lock, []{ return num_done == num_submitted; });
}
