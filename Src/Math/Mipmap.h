#pragma once
#include "Math.h"
#include "Vector4.h"

namespace Mipmap {
	void downsample(int width_src, int height_src, int width_dst, int height_dst, const Vector4 texture_src[], Vector4 texture_dst[], Vector4 temp[]);
}