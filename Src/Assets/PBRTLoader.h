#pragma once
#include "Core/String.h"

struct Scene;

namespace PBRTLoader {
	void load(const String & filename, Allocator * allocator, Scene & scene);
}
