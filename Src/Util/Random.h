#pragma once

namespace Random {
	void init();
	void init(unsigned seed);

	unsigned get_value();
	unsigned get_value(unsigned max);
}
