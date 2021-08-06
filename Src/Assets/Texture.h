#pragma once
#include <cuda.h>

#include "../CUDA_Source/Common.h"

struct Scene;

struct Texture {
	enum struct Format {
		BC1,
		BC2,
		BC3,
		RGBA
	};

	const unsigned char * data = nullptr;
	
	Format format = Format::RGBA;
	
	int channels;
	int width, height;

	int         mip_levels;
	const int * mip_offsets; // Offsets in bytes

	void free();

	CUarray_format       get_cuda_array_format()         const;
	CUresourceViewFormat get_cuda_resource_view_format() const;

	int get_cuda_resource_view_width()  const;
	int get_cuda_resource_view_height() const;

	int get_width_in_bytes(int mip_level = 0) const;
};

struct TextureHandle { int handle = INVALID; };
