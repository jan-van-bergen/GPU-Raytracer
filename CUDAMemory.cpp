#include "CUDAMemory.h"

#include <GL/glew.h>
#include <cudaGL.h>

CUarray CUDAMemory::create_array(int width, int height, int channels, CUarray_format format) {
	CUDA_ARRAY_DESCRIPTOR desc;
	desc.Width  = width;
	desc.Height = height;
	desc.NumChannels = channels;
	desc.Format = format;
		
	CUarray array;
	CUDACALL(cuArrayCreate(&array, &desc));

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
