#include "Mutex.h"

#include <mutex>

struct Impl { std::mutex mutex; };

Mutex::Mutex() {
	impl = OwnPtr<Impl>::make();
}

Mutex::~Mutex() { }

void Mutex::lock()   { impl->mutex.lock(); }
void Mutex::unlock() { impl->mutex.unlock(); }
