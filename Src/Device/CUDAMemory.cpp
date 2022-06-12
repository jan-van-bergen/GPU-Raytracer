#include "CUDAMemory.h"

#include <GL/glew.h>
#include <cudaGL.h>

CUarray CUDAMemory::create_array(int width, int height, int channels, CUarray_format format) {
	CUDA_ARRAY_DESCRIPTOR desc = { };
	desc.Width       = width;
	desc.Height      = height;
	desc.NumChannels = channels;
	desc.Format      = format;

	CUarray array;
	CUDACALL(cuArrayCreate(&array, &desc));

	return array;
}

CUarray CUDAMemory::create_array_3d(int width, int height, int depth, int channels, CUarray_format format) {
	CUDA_ARRAY3D_DESCRIPTOR desc = { };
	desc.Width       = width;
	desc.Height      = height;
	desc.Depth       = depth;
	desc.NumChannels = channels;
	desc.Format      = format;

	CUarray array;
	CUDACALL(cuArray3DCreate(&array, &desc));

	return array;
}

CUmipmappedArray CUDAMemory::create_array_mipmap(int width, int height, int channels, CUarray_format format, int level_count) {
	CUDA_ARRAY3D_DESCRIPTOR desc = { };
	desc.Width       = width;
	desc.Height      = height;
	desc.Depth       = 0;
	desc.NumChannels = channels;
	desc.Format      = format;
	desc.Flags       = 0;

	CUmipmappedArray mipmap;
	CUDACALL(cuMipmappedArrayCreate(&mipmap, &desc, level_count));

	return mipmap;
}

void CUDAMemory::free_array(CUarray array) {
	CUDACALL(cuArrayDestroy(array));
}

void CUDAMemory::free_array(CUmipmappedArray array) {
	CUDACALL(cuMipmappedArrayDestroy(array));
}

// Copies data from the Host Texture to the Device Array
void CUDAMemory::copy_array(CUarray array, int width_in_bytes, int height, const void * data) {
	CUDA_MEMCPY2D copy = { };
	copy.srcMemoryType = CU_MEMORYTYPE_HOST;
	copy.srcHost       = data;
	copy.srcPitch      = width_in_bytes;
	copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copy.dstArray      = array;
	copy.WidthInBytes  = width_in_bytes;
	copy.Height        = height;

	CUDACALL(cuMemcpy2D(&copy));
}

// Copies data from the Host Texture to the Device Array
void CUDAMemory::copy_array_3d(CUarray array, int width_in_bytes, int height, int depth, const void * data) {
	CUDA_MEMCPY3D copy = { };
	copy.srcMemoryType = CU_MEMORYTYPE_HOST;
	copy.srcHost       = data;
	copy.srcPitch      = width_in_bytes;
	copy.srcHeight     = height;
	copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copy.dstArray      = array;
	copy.WidthInBytes  = width_in_bytes;
	copy.Height        = height;
	copy.Depth         = depth;

	CUDACALL(cuMemcpy3D(&copy));
}

// Copies data from the Host Texture to the Device Array
void CUDAMemory::copy_array(CUarray array, int width_in_bytes, int height, CUdeviceptr data) {
	CUDA_MEMCPY2D copy = { };
	copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copy.srcDevice     = data;
	copy.srcPitch      = width_in_bytes;
	copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copy.dstArray      = array;
	copy.WidthInBytes  = width_in_bytes;
	copy.Height        = height;

	CUDACALL(cuMemcpy2D(&copy));
}

// Copies data from the Host Texture to the Device Array
void CUDAMemory::copy_array_3d(CUarray array, int width_in_bytes, int height, int depth, CUdeviceptr data) {
	CUDA_MEMCPY3D copy = { };
	copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
	copy.srcDevice     = data;
	copy.srcPitch      = width_in_bytes;
	copy.srcHeight     = height;
	copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
	copy.dstArray      = array;
	copy.WidthInBytes  = width_in_bytes;
	copy.Height        = height;
	copy.Depth         = depth;

	CUDACALL(cuMemcpy3D(&copy));
}

CUtexObject CUDAMemory::create_texture(CUarray array, CUfilter_mode filter, CUaddress_mode address_mode) {
	CUDA_RESOURCE_DESC res_desc = { };
	res_desc.resType = CUresourcetype::CU_RESOURCE_TYPE_ARRAY;
	res_desc.res.array.hArray = array;

	CUDA_TEXTURE_DESC tex_desc = { };
	tex_desc.addressMode[0] = address_mode;
	tex_desc.addressMode[1] = address_mode;
	tex_desc.addressMode[2] = address_mode;
	tex_desc.filterMode = filter;
	tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;

	CUtexObject tex_object = { };
	CUDACALL(cuTexObjectCreate(&tex_object, &res_desc, &tex_desc, nullptr));

	return tex_object;
}

CUsurfObject CUDAMemory::create_surface(CUarray array) {
	CUDA_RESOURCE_DESC desc = { };
	desc.resType = CU_RESOURCE_TYPE_ARRAY;
	desc.res.array.hArray = array;

	CUsurfObject surf_object = { };
	CUDACALL(cuSurfObjectCreate(&surf_object, &desc));

	return surf_object;
}

void CUDAMemory::free_texture(CUtexObject texture) {
	CUDACALL(cuTexObjectDestroy(texture));
}

void CUDAMemory::free_surface(CUsurfObject surface) {
	CUDACALL(cuSurfObjectDestroy(surface));
}

CUgraphicsResource CUDAMemory::resource_register(unsigned gl_texture, unsigned flags) {
	CUgraphicsResource resource;
	CUDACALL(cuGraphicsGLRegisterImage(&resource, gl_texture, GL_TEXTURE_2D, flags));

	return resource;
}

void CUDAMemory::resource_unregister(CUgraphicsResource resource) {
	CUDACALL(cuGraphicsUnregisterResource(resource));
}

CUarray CUDAMemory::resource_get_array(CUgraphicsResource resource) {
	CUDACALL(cuGraphicsMapResources(1, &resource, 0));

	CUarray result;
	CUDACALL(cuGraphicsSubResourceGetMappedArray(&result, resource, 0, 0));

	CUDACALL(cuGraphicsUnmapResources(1, &resource, 0));

	return result;
}
