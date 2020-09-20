#pragma once
#include <vector>

#include <cuda.h>

struct Texture {
	enum class Format {
		BC1,
		BC2,
		BC3,
		RGBA
	};

	const unsigned char * data;
	
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

	int get_width_in_bytes() const;

	static int load(const char * file_path);

	static void wait_until_textures_loaded();

	inline static std::vector<Texture> textures;
};
