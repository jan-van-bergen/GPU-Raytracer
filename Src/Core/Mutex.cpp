#include "Mutex.h"

#include <mutex>

struct Impl { std::mutex mutex; };

void Mutex::init() {
	impl = new Impl();
}

void Mutex::free() {
	delete impl;
}

void Mutex::lock() {
	impl->mutex.lock();
}

void Mutex::unlock() {
	impl->mutex.unlock();
}
