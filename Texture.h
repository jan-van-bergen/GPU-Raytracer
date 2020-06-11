#pragma once
#include <vector>

#include <cuda.h>

#include "CUDA_Source/Common.h"

struct Texture {
	enum class Format {
		RGBA_COMPRESSED_BC1,
		RGBA_COMPRESSED_BC2,
		RGBA_COMPRESSED_BC3,
		RGBA
	};

	unsigned char * data = nullptr;

	Format format = Format::RGBA;

	int channels;
	int width, height;

	CUarray_format       get_cuda_array_format() const;
	CUresourceViewFormat get_cuda_resource_view_format() const;

	int get_cuda_resource_view_width()  const;
	int get_cuda_resource_view_height() const;

	int get_width_in_bytes() const;

	static int load(const char * file_path);

	static std::vector<Texture> textures;
};
