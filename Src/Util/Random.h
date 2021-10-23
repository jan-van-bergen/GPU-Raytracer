#pragma once
#include <stdint.h>

// Based on: https://github.com/imneme/pcg-c
struct RNG {
	uint64_t state;

	void init(uint64_t seed) {
		constexpr uint32_t MUL32 = 747796405u;
		constexpr uint32_t INC32 = 2891336453u;
		state = (seed + INC32) * MUL32 + INC32;
	}

	uint32_t get_uint32() {
		uint32_t x = ((state >> 18u) ^ state) >> 27u;
		uint32_t r = state >> 59u;
		state  = state * 6364136223846793005ull + 1;
		return (x >> r) | (x << ((~r + 1) & 31));
	}

	float get_float() {
		return get_uint32() / float(0xffffffff);
	}
};

namespace Util {
	// Based on Knuth's selection sampling algorithm
	template<typename T>
	void sample(const T * src_first, const T * src_last, T * dst_first, T * dst_last, RNG & rng) {
		for (const T * it = src_first; it < src_last; it++) {
			float u = float(src_last - it) * rng.get_float();
			if (u < dst_last - dst_first) {
				*dst_first++ = *it;

				if (dst_first == dst_last) {
					break; // Done
				}
			}
		}
	}
}
