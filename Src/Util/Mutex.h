#pragma once

// Wrapper around std::mutex
// Avoids having to include <mutex> everywhere, which slows down compile times
struct Mutex {
	struct Impl * impl;

	void init();
	void free();

	void lock();
	void unlock();
};
