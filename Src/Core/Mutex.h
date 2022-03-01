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

struct MutexLock {
	Mutex & mutex;

	MutexLock(Mutex & mutex) : mutex(mutex) {
		mutex.lock();
	}

	NON_COPYABLE(MutexLock);
	NON_MOVEABLE(MutexLock);

	~MutexLock() {
		mutex.unlock();
	}
};
