#pragma once
#include "Core/Allocators/Allocator.h"

namespace Args {
	void parse(int num_args, char ** args, Allocator * allocator);
}
