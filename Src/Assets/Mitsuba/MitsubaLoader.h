#pragma once
#include "Util/StringUtil.h"

struct Scene;

namespace MitsubaLoader {
	void load(const String & filename, Scene & scene);
}
