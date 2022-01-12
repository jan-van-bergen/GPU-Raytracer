#pragma once
#include "OwnPtr.h"

// Wrapper around std::mutex
// Avoids having to include <mutex> everywhere, which slows down compile times
struct Mutex {
	OwnPtr<struct Impl> impl;

	Mutex();
	~Mutex();

	void lock();
	void unlock();
};
