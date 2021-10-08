#include "ThreadPool.h"

void ThreadPool::init(int thread_count) {
	this->thread_count = thread_count;
	threads = new std::thread[thread_count];

	for (int i = 0; i < thread_count; i++) {
		threads[i] = std::thread([this]() {
			while (true) {
				Work work;
				{
					std::unique_lock<std::mutex> lock(signal_submit.mutex);
					signal_submit.condition.wait(lock, [this]{ return !work_queue.empty() || is_done; });

					if (is_done) return;

					work = std::move(work_queue.pop());
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

	for (int i = 0; i < thread_count; i++) {
		threads[i].join();
	}
	delete [] threads;
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
	signal_done.condition.wait(lock, [this]{ return num_done == num_submitted; });
}
