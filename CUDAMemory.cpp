#include "CUDAMemory.h"

static int get_array_format_size(CUarray_format format) {
	int format_size;
	switch (format) {
		case CU_AD_FORMAT_UNSIGNED_INT8:  format_size = 1; break;
		case CU_AD_FORMAT_UNSIGNED_INT16: format_size = 2; break;
		case CU_AD_FORMAT_UNSIGNED_INT32: format_size = 4; break;
		case CU_AD_FORMAT_SIGNED_INT8:	  format_size = 1; break;
		case CU_AD_FORMAT_SIGNED_INT16:   format_size = 2; break;
		case CU_AD_FORMAT_SIGNED_INT32:   format_size = 4; break;
		case CU_AD_FORMAT_HALF:           format_size = 2; break;
		case CU_AD_FORMAT_FLOAT:          format_size = 4; break;

		default: abort();
	}

	return format_size;
}

CUarray CUDAMemory::create_array(int width, int height, int channels, CUarray_format format) {
	CUDA_ARRAY_DESCRIPTOR desc;
	desc.Width  = width;
	desc.Height = height;
	desc.NumChannels = channels;
	desc.Format = format;
		
	CUarray array;
	CUDACALL(cuArrayCreate(&array, &desc));

	memory_usage += width * height * channels * get_array_format_size(format);

	return array;
}

CUarray CUDAMemory::create_array3d(int width, int height, int depth, int channels, CUarray_format format, unsigned flags) {
	CUDA_ARRAY3D_DESCRIPTOR desc;
	desc.Width  = width;
	desc.Height = height;
	desc.Depth  = depth;
	desc.NumChannels = channels;
	desc.Format = format;
	desc.Flags  = flags;
		
	CUarray array;
	CUDACALL(cuArray3DCreate(&array, &desc));
	
	memory_usage += width * height * depth * channels * get_array_format_size(format);

	return array;
}

// Copies data from the Host Texture to the Device Array
void CUDAMemory::copy_array(CUarray array, int width_in_bytes, int height, const void * data) {
	CUDA_MEMCPY2D copy = { };
	copy.srcMemoryType = CU_MEMORYTYPE_HOST;
	copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copy.srcHost  = data;
	copy.dstArray = array;
	copy.srcPitch = width_in_bytes;
	copy.WidthInBytes = copy.srcPitch;
	copy.Height       = height;

	CUDACALL(cuMemcpy2D(&copy));
}
