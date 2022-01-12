#pragma once
#include <chrono>

#include "IO.h"

// Timer that records the time between its construction and destruction
struct ScopeTimer {
private:
	const char * name;
	std::chrono::high_resolution_clock::time_point start_time;

public:
	inline ScopeTimer(const char * name) : name(name) {
		start_time = std::chrono::high_resolution_clock::now();
	}

	inline ~ScopeTimer() {
		std::chrono::time_point stop_time = std::chrono::high_resolution_clock::now();
		unsigned long long      duration  = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

		if (duration >= 60000000) {
			IO::print("{} took: {} s ({} min)\n"sv, name, duration / 1000000, duration / 60000000);
		} else if (duration >= 1000000) {
			IO::print("{} took: {} us ({} s)\n"sv, name, duration, duration / 1000000);
		} else if (duration >= 1000) {
			IO::print("{} took: {} us ({} ms)\n"sv, name, duration, duration / 1000);
		} else {
			IO::print("{} took: {} us\n"sv, name, duration);
		}
	}
};
