#pragma once
#include <random>

#include "../CUDA_Source/Common.h"

#include "Util/Util.h"

namespace PMJ {
	struct Point {
		unsigned x, y;
	};
	extern Point samples[];

	// Based on: https://github.com/blender/blender/blob/b4c9f88cbede8bc27d4d869232fd4a8f59e39f40/intern/cycles/render/jitter.cpp#L243
	inline void shuffle(int sequence_index) {
		std::default_random_engine rng;
		rng.seed(sequence_index);

		std::uniform_real_distribution<float> dist(0.0f, 1.0f);

		constexpr int odd [8] = { 0, 1, 4, 5, 10, 11, 14, 15 };
		constexpr int even[8] = { 2, 3, 6, 7, 8,  9,  12, 13 };

		// Shuffle sequence in blocks of 16
		Point * sequence = samples + sequence_index * PMJ_NUM_SAMPLES_PER_SEQUENCE;

		for (int j = 0; j < PMJ_NUM_SAMPLES_PER_SEQUENCE / 16; j++) {
			for (int i = 0; i < 8; i++) {
				int other = (int)(dist(rng) * (8.0f - i) + i);
				Util::swap(
					sequence[odd[other] + j * 16],
					sequence[odd[i]     + j * 16]
				);
			}
			for (int i = 0; i < 8; i++) {
				int other = (int)(dist(rng) * (8.0f - i) + i);
				Util::swap(
					sequence[even[other] + j * 16],
					sequence[even[i]     + j * 16]
				);
			}
		}
	}
}
