#pragma once
#include "Math.h"
#include "Vector4.h"

namespace Mipmap {
	/*
		Mipmap filter code based on http://number-none.com/product/Mipmapping,%20Part%201/index.html and https://github.com/castano/nvidia-texture-tools
	*/

	struct FilterBox {
		static constexpr float width = 0.5f;

		static float eval(float x) {
			if (fabsf(x) <= width) {
				return 1.0f;
			} else {
				return 0.0f;
			}
		}
	};

	struct FilterLanczos {
		static constexpr float width = 3.0f;

		static float eval(float x) {
			if (fabsf(x) < width) {
				return Math::sincf(PI * x) * Math::sincf(PI * x / width);
			} else {
				return 0.0f;
			}
		}
	};

	struct FilterKaiser {
		static constexpr float width   = 7.0f;
		static constexpr float alpha   = 4.0f;
		static constexpr float stretch = 1.0f;

		static float eval(float x) {
			float t  = x / width;
			float t2 = t * t;

			if (t2 < 1.0f) {
				return Math::sincf(PI * x * stretch) * Math::bessel_0(alpha * sqrtf(1.0f - t2)) / Math::bessel_0(alpha);
			} else {
				return 0.0f;
			}
		}
	};
	
#if MIPMAP_DOWNSAMPLE_FILTER == MIPMAP_DOWNSAMPLE_FILTER_BOX
	typedef FilterBox Filter;
#elif MIPMAP_DOWNSAMPLE_FILTER == MIPMAP_DOWNSAMPLE_FILTER_LANCZOS
	typedef FilterLanczos Filter;
#elif MIPMAP_DOWNSAMPLE_FILTER == MIPMAP_DOWNSAMPLE_FILTER_KAISER
	typedef FilterKaiser Filter;
#endif

	void downsample(int width_src, int height_src, int width_dst, int height_dst, const Vector4 texture_src[], Vector4 texture_dst[], Vector4 temp[]);
}
