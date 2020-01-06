#pragma once

#include <cudaGL.h>

#include "CUDACall.h"

namespace CUDAContext {
	inline static int compute_capability = -1;

	// Creates a new CUDA Context
	inline void init() {
		CUDACALL(cuInit(0));

		int device_count;
		CUDACALL(cuDeviceGetCount(&device_count));

		unsigned gl_device_count;
		CUdevice * devices = new CUdevice[device_count];

		CUDACALL(cuGLGetDevices(&gl_device_count, devices, device_count, CU_GL_DEVICE_LIST_ALL));
	
		CUdevice device = devices[0];

		CUcontext context;
		CUDACALL(cuGLCtxCreate(&context, 0, device));

		delete [] devices;
	
		int major, minor;
		CUDACALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
		CUDACALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
		compute_capability = major * 10 + minor;

	}

	// Creates a CUDA Array that is mapped to the given GL Texture handle
	inline CUarray map_gl_texture(unsigned gl_texture) {
		CUarray result;

		CUgraphicsResource cuda_frame_buffer_handle; 
		CUDACALL(cuGraphicsGLRegisterImage(&cuda_frame_buffer_handle, gl_texture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));
		CUDACALL(cuGraphicsMapResources(1, &cuda_frame_buffer_handle, 0));

		CUDACALL(cuGraphicsSubResourceGetMappedArray(&result, cuda_frame_buffer_handle, 0, 0));
                
		CUDACALL(cuGraphicsUnmapResources(1, &cuda_frame_buffer_handle, 0));

		return result;
	}
}
