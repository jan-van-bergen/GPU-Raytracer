#pragma once
#include <stdint.h>

#include "Core/Assertion.h"

// Based on: https://github.com/imneme/pcg-c
struct RNG {
	uint64_t state;

	RNG(uint64_t seed) {
		constexpr uint32_t MUL32 = 747796405u;
		constexpr uint32_t INC32 = 2891336453u;
		state = (seed + INC32) * MUL32 + INC32;
	}

	uint32_t get_uint32() {
		uint32_t x = uint32_t(((state >> 18u) ^ state) >> 27u);
		uint32_t r = state >> 59u;
		state  = state * 6364136223846793005ull + 1;
		return (x >> r) | (x << ((~r + 1) & 31));
	}

	// Based on: https://www.pcg-random.org/posts/bounded-rands.html
	uint32_t get_uint32(uint32_t max) {
		uint32_t x = get_uint32();
		uint64_t m = uint64_t(x) * uint64_t(max);
		uint32_t l = uint32_t(m);
		if (l < max) {
			uint32_t t = ~max + 1;
			if (t >= max) {
				t -= max;
				if (t >= max) t %= max;
			}
			while (l < t) {
				x = get_uint32();
				m = uint64_t(x) * uint64_t(max);
				l = uint32_t(m);
			}
		}
		return m >> 32;
	}

	uint32_t get_uint32(uint32_t min, uint32_t max) {
		return min + get_uint32(max - min);
	}

	float get_float() {
		return get_uint32() / float(0xffffffff);
	}
};

namespace Random {
	// Based on Knuth's selection sampling algorithm
	template<typename T>
	void sample(const T * src_first, const T * src_last, T * dst_first, T * dst_last, RNG & rng) {
		for (const T * it = src_first; it < src_last; it++) {
			if (rng.get_uint32(src_last - it) < dst_last - dst_first) {
				*dst_first++ = *it;

				if (dst_first == dst_last) {
					return; // Done
				}
			}
		}
		ASSERT_UNREACHABLE();
	}
}
