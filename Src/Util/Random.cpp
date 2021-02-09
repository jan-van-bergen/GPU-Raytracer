#include "Random.h"

#include <random>

static std::default_random_engine              engine;
static std::uniform_int_distribution<unsigned> uniform_dist;

void Random::init() {
	init(std::random_device()());
}

void Random::init(unsigned seed) {
	engine       = std::default_random_engine(seed);
	uniform_dist = std::uniform_int_distribution<unsigned>(0, UINT32_MAX);
}

unsigned Random::get_value() {
	return uniform_dist(engine);
}

unsigned Random::get_value(unsigned max) {
	 return std::uniform_int_distribution<unsigned>(0, max)(engine);
}
