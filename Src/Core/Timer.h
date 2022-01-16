#pragma once
#include <chrono>

#include "IO.h"

// Manual Timer
struct Timer {
	std::chrono::high_resolution_clock::time_point start_time;

	void start() {
		start_time = std::chrono::high_resolution_clock::now();
	}

	size_t stop() {
		std::chrono::time_point stop_time = std::chrono::high_resolution_clock::now();
		return std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time).count();
	}

	static void print_named_duration(StringView name, size_t duration) {
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

// Timer that records the time between its construction and destruction
struct ScopeTimer {
	StringView name;
	Timer      timer;

	ScopeTimer(StringView name) : name(name) {
		timer.start();
	}

	~ScopeTimer() {
		size_t duration = timer.stop();
		Timer::print_named_duration(name, duration);
	}
};
