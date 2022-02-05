#include "Texture.h"

#include "Math/Math.h"
#include "Math/Vector4.h"

CUarray_format Texture::get_cuda_array_format() const {
	switch (format) {
		case Format::BC1:  return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::BC2:  return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::BC3:  return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::RGBA: return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT8;

		default: ASSERT(false);
	}
}

CUresourceViewFormat Texture::get_cuda_resource_view_format() const {
	switch (format) {
		case Texture::Format::BC1:  return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
		case Texture::Format::BC2:  return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
		case Texture::Format::BC3:  return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
		case Texture::Format::RGBA: return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UINT_4X8;

		default: ASSERT(false);
	}
}

int Texture::get_cuda_resource_view_width() const {
	if (format == Format::RGBA) {
		return width;
	} else {
		return width * 4;
	}
}

int Texture::get_cuda_resource_view_height() const {
	if (format == Format::RGBA) {
		return height;
	} else {
		return height * 4;
	}
}

int Texture::get_width_in_bytes(int mip_level) const {
	int level_width = Math::max(width >> mip_level, 1);

	if (format == Format::RGBA) {
		return level_width * sizeof(unsigned);
	} else {
		return level_width * channels * 4;
	}
}
