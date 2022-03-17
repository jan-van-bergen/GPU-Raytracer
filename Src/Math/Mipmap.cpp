#include "Mipmap.h"

#include <string.h>
#include <stdlib.h>

#include "Config.h"

#include "Core/Allocators/StackAllocator.h"

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

template<typename Filter>
static float filter_sample_box(float x, float scale) {
	constexpr int   SAMPLE_COUNT     = 32;
	constexpr float SAMPLE_COUNT_INV = 1.0f / float(SAMPLE_COUNT);

	float sample = 0.5f;
	float sum    = 0.0f;

	for (int i = 0; i < SAMPLE_COUNT; i++, sample += 1.0f) {
		float p = (x + sample * SAMPLE_COUNT_INV) * scale;

		sum += Filter::eval(p);
	}

	return sum * SAMPLE_COUNT_INV;
}

template<typename Filter>
void downsample_impl(int width_src, int height_src, int width_dst, int height_dst, const Vector4 texture_src[], Vector4 texture_dst[], Vector4 temp[]) {
	float scale_x = float(width_dst)  / float(width_src);
	float scale_y = float(height_dst) / float(height_src);

	ASSERT(scale_x <= 1.0f && scale_y <= 1.0f);

	float inv_scale_x = 1.0f / scale_x;
	float inv_scale_y = 1.0f / scale_y;

	float filter_width_x = Filter::width * inv_scale_x;
	float filter_width_y = Filter::width * inv_scale_y;

	int window_size_x = int(ceilf(filter_width_x * 2.0f)) + 1;
	int window_size_y = int(ceilf(filter_width_y * 2.0f)) + 1;

	StackAllocator<KILOBYTES(1)> allocator;
	Array<float> kernel_x(window_size_x, &allocator);
	Array<float> kernel_y(window_size_y, &allocator);
	memset(kernel_x.data(), 0, window_size_x * sizeof(float));
	memset(kernel_y.data(), 0, window_size_y * sizeof(float));

	float sum_x = 0.0f;
	float sum_y = 0.0f;

	// Fill horizontal kernel
	for (int x = 0; x < window_size_x; x++) {
		float sample = filter_sample_box<Filter>(float(x - window_size_x / 2), scale_x);

		kernel_x[x] = sample;
		sum_x += sample;
	}

	// Fill vertical kernel
	for (int y = 0; y < window_size_y; y++) {
		float sample = filter_sample_box<Filter>(float(y - window_size_y / 2), scale_y);

		kernel_y[y] = sample;
		sum_y += sample;
	}

	// Normalize kernels
	for (int x = 0; x < window_size_x; x++) kernel_x[x] /= sum_x;
	for (int y = 0; y < window_size_y; y++) kernel_y[y] /= sum_y;

	// Apply horizontal kernel
	for (int y = 0; y < height_src; y++) {
		for (int x = 0; x < width_dst; x++) {
			float center = (float(x) + 0.5f) * inv_scale_x;

			int left = int(floorf(center - filter_width_x));

			Vector4 sum = Vector4(0.0f);

			for (int i = 0; i < window_size_x; i++) {
				int index = Math::clamp(left + i, 0, width_src - 1) + y * width_src;

				sum += kernel_x[i] * texture_src[index];
			}

			temp[x * height_src + y] = sum;
		}
	}

	// Apply vertical kernel
	for (int x = 0; x < width_dst; x++) {
		for (int y = 0; y < height_dst; y++) {
			float center = (float(y) + 0.5f) * inv_scale_y;

			int top = int(floorf(center - filter_width_y));

			Vector4 sum = Vector4(0.0f);

			for (int i = 0; i < window_size_y; i++) {
				int index = x * height_src + Math::clamp(top + i, 0, height_src - 1);

				sum += kernel_y[i] * temp[index];
			}

			texture_dst[x + y * width_dst] = sum;
		}
	}
}

void Mipmap::downsample(int width_src, int height_src, int width_dst, int height_dst, const Vector4 texture_src[], Vector4 texture_dst[], Vector4 temp[]) {
	switch (cpu_config.mipmap_filter) {
		case MipmapFilterType::BOX:     downsample_impl<FilterBox>    (width_src, height_src, width_dst, height_dst, texture_src, texture_dst, temp); break;
		case MipmapFilterType::LANCZOS: downsample_impl<FilterLanczos>(width_src, height_src, width_dst, height_dst, texture_src, texture_dst, temp); break;
		case MipmapFilterType::KAISER:  downsample_impl<FilterKaiser> (width_src, height_src, width_dst, height_dst, texture_src, texture_dst, temp); break;
		default: ASSERT_UNREACHABLE();
	}
}
