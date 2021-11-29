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

		size_t duration_in_ms  = (stop_time - start_time) * 1000 / CLOCKS_PER_SEC;
		size_t duration_in_s   = duration_in_ms / 1000;
		size_t duration_in_min = duration_in_s  / 60;

		if (duration_in_min > 0) {
			printf("%s took: %llu s (%llu min)\n", name, duration_in_s, duration_in_min);
		} else if (duration_in_s > 0) {
			printf("%s took: %llu ms (%llu s)\n", name, duration_in_ms, duration_in_s);
		} else {
			printf("%s took: %llu ms\n", name, duration_in_ms);
		}
	}
};
