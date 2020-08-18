#include "Random.h"

#include <random>

static std::default_random_engine              engine;
static std::uniform_int_distribution<unsigned> uniform_dist;

void Random::init() {
	engine       = std::default_random_engine(std::random_device()());
	uniform_dist = std::uniform_int_distribution<unsigned>(0, UINT32_MAX);
}

void Random::init(unsigned seed) {
	engine       = std::default_random_engine(seed);
	uniform_dist = std::uniform_int_distribution<unsigned>(0, UINT32_MAX);
}

unsigned Random::get_value() {
	return uniform_dist(engine);
}
