#pragma once
#include <cstdio>

#define ASSERT(assertion)                                                          \
	do {                                                                           \
		if (!(assertion)) {                                                        \
			printf("%s:%i: ASSERT(" #assertion ") failed!\n", __FILE__, __LINE__); \
			__debugbreak();                                                        \
		}                                                                          \
	} while(false)

#define ASSERT_UNREACHABLE() abort()
