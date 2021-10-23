#pragma once

namespace PMJ {
	struct Point {
		unsigned x, y;
	};
	extern Point samples[];

	void shuffle(int sequence_index);
}
