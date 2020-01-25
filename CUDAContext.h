#pragma once
#include <GL/glew.h>
#include <cudaGL.h>

#include "CUDACall.h"

namespace CUDAContext {
	inline int compute_capability = -1;

	// Creates a new CUDA Context
	inline void init() {
		CUDACALL(cuInit(0));

		int device_count;
		CUDACALL(cuDeviceGetCount(&device_count));

		CUdevice * devices = new CUdevice[device_count];

		unsigned gl_device_count;
		CUDACALL(cuGLGetDevices(&gl_device_count, devices, device_count, CU_GL_DEVICE_LIST_ALL));
	
		CUdevice best_device;
		int      best_compute_capability = 0;

		for (int i = 0; i < gl_device_count; i++) {
			int major, minor;
			CUDACALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices[i]));
			CUDACALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices[i]));

			int	device_compute_capability = major * 10 + minor;
			if (device_compute_capability > best_compute_capability) {
				best_device = devices[i];
				best_compute_capability = device_compute_capability;
			}
		}

		compute_capability = best_compute_capability;

		delete [] devices;

		CUcontext context;
		CUDACALL(cuGLCtxCreate(&context, 0, best_device));
	}

	// Creates a CUDA Array that is mapped to the given GL Texture handle
	inline CUarray map_gl_texture(unsigned gl_texture) {
		CUarray result;

		CUgraphicsResource cuda_resource; 
		CUDACALL(cuGraphicsGLRegisterImage(&cuda_resource, gl_texture, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));
		CUDACALL(cuGraphicsMapResources(1, &cuda_resource, 0));

		CUDACALL(cuGraphicsSubResourceGetMappedArray(&result, cuda_resource, 0, 0));
                
		CUDACALL(cuGraphicsUnmapResources(1, &cuda_resource, 0));

		return result;
	}
}
