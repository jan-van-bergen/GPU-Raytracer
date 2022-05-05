#pragma once
#include "Core/Function.h"

namespace ThreadPool {
	using Work = Function<void()>;

	void init();
	void init(int thread_count);
	void free();

	void submit(Work && work);

	void sync();
};
