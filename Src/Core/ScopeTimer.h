#pragma once
#include <chrono>

#include "IO.h"

// Timer that records the time between its construction and destruction
struct ScopeTimer {
	StringView name;
	std::chrono::high_resolution_clock::time_point start_time;

	inline ScopeTimer(StringView name) : name(name) {
		start_time = std::chrono::high_resolution_clock::now();
	}

	inline ~ScopeTimer() {
		std::chrono::time_point stop_time = std::chrono::high_resolution_clock::now();
		size_t                  duration  = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();

		if (duration >= 60000000) {
			IO::print("{} took: {} s ({} min)\n"_sv, name, duration / 1000000, duration / 60000000);
		} else if (duration >= 1000000) {
			IO::print("{} took: {} us ({} s)\n"_sv, name, duration, duration / 1000000);
		} else if (duration >= 1000) {
			IO::print("{} took: {} us ({} ms)\n"_sv, name, duration, duration / 1000);
		} else {
			IO::print("{} took: {} us\n"_sv, name, duration);
		}
	}
};
