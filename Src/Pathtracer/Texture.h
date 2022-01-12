#pragma once
#include <cuda.h>

#include "Core/Array.h"
#include "Core/String.h"

#include "CUDA/Common.h"

struct Scene;

struct Texture {
	String name = "Texture";

	Array<unsigned char> data;

	enum struct Format {
		BC1,
		BC2,
		BC3,
		RGBA
	} format = Format::RGBA;

	int channels;
	int width, height;

	Array<int> mip_offsets; // Offsets in bytes

	CUarray_format       get_cuda_array_format()         const;
	CUresourceViewFormat get_cuda_resource_view_format() const;

	int get_cuda_resource_view_width()  const;
	int get_cuda_resource_view_height() const;

	int get_width_in_bytes(int mip_level = 0) const;

	inline int mip_levels() const { return mip_offsets.size(); }
};

struct TextureHandle { int handle = INVALID; };
