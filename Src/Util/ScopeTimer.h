#pragma once
#include <time.h>
#include <stdio.h>

// Timer that records the time between its construction and destruction
struct ScopeTimer {
private:
	const char * name;
	clock_t start_time;

public:
	inline ScopeTimer(const char * name) : name(name) {
		start_time = clock();
	}

	inline ~ScopeTimer() {
		clock_t stop_time = clock();
		size_t  duration  = (stop_time - start_time) * 1000 / CLOCKS_PER_SEC;

		if (duration >= 60'000'000) {
			printf("%s took: %llu s (%llu min)\n", name, duration / 1'000'000, duration / 60'000'000);
		} else if (duration >= 1'000'000) {
			printf("%s took: %llu us (%llu s)\n", name, duration, duration / 1'000'000);
		} else if (duration >= 1'000) {
			printf("%s took: %llu us (%llu ms)\n", name, duration, duration / 1'000);
		} else {
			printf("%s took: %llu us\n", name, duration);
		}
	}
};
