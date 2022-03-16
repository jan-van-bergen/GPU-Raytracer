#include "CUDAContext.h"

#include <malloc.h>

#include <GL/glew.h>
#include <cudaGL.h>

#include "Core/IO.h"
#include "Core/Allocators/StackAllocator.h"

static CUdevice  device;
static CUcontext context;

static int device_get_attribute(CUdevice_attribute attribute, CUdevice cuda_device = device) {
	int result;
	CUDACALL(cuDeviceGetAttribute(&result, attribute, cuda_device));

	return result;
}

void CUDAContext::init() {
	CUDACALL(cuInit(0));

	int device_count;
	CUDACALL(cuDeviceGetCount(&device_count));

	if (device_count == 0) {
		IO::print("ERROR: No suitable Device found!\n"_sv);
		IO::exit(1);
	}

	StackAllocator<128> allocator;
	Array<CUdevice> devices(device_count, &allocator);

	unsigned gl_device_count;
	CUDACALL(cuGLGetDevices(&gl_device_count, devices.data(), device_count, CU_GL_DEVICE_LIST_ALL));

	if (gl_device_count == 0) {
		IO::print("ERROR: No suitable GL Device found!\n"_sv);
		IO::exit(1);
	}

	CUdevice best_device;
	int      best_compute_capability = 0;

	for (unsigned i = 0; i < gl_device_count; i++) {
		int major = device_get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, devices[i]);
		int minor = device_get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, devices[i]);

		int	device_compute_capability = major * 10 + minor;
		if (device_compute_capability > best_compute_capability) {
			best_device             = devices[i];
			best_compute_capability = device_compute_capability;
		}
	}

	compute_capability = best_compute_capability;

	device = best_device;

	CUDACALL(cuCtxCreate(&context, 0, device));

	CUfunc_cache   config_cache;
	CUsharedconfig config_shared;
	CUDACALL(cuCtxGetCacheConfig    (&config_cache));
	CUDACALL(cuCtxGetSharedMemConfig(&config_shared));

	size_t bytes_free = 0;
	CUDACALL(cuMemGetInfo(&bytes_free, &total_memory));

	int driver_version = 0;
	CUDACALL(cuDriverGetVersion(&driver_version));

	IO::print("CUDA Info:\n"_sv);
	IO::print("CUDA Version: {}.{}\n"_sv, driver_version / 1000, (driver_version % 1000) / 10);
	IO::print("Compute Capability: {}\n"_sv, compute_capability);
	IO::print("Memory available: {} MB\n"_sv, total_memory >> 20);

	switch (config_cache) {
		case CU_FUNC_CACHE_PREFER_NONE:   IO::print("Cache Config: Prefer None\n"_sv);   break;
		case CU_FUNC_CACHE_PREFER_SHARED: IO::print("Cache Config: Prefer Shared\n"_sv); break;
		case CU_FUNC_CACHE_PREFER_L1:     IO::print("Cache Config: Prefer L1\n"_sv);     break;
		case CU_FUNC_CACHE_PREFER_EQUAL:  IO::print("Cache Config: Prefer Equal\n"_sv);  break;
	}

	switch (config_shared) {
		case CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE:    IO::print("Shared Memory Config: Default\n"_sv); break;
		case CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE:  IO::print("Shared Memory Config: 4 Bytes\n"_sv); break;
		case CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE: IO::print("Shared Memory Config: 8 Bytes\n"_sv); break;
	}

	IO::print('\n');
}

void CUDAContext::free() {
	CUDACALL(cuCtxDestroy(context));
	CUDACALL(cuDevicePrimaryCtxReset(device));
}

size_t CUDAContext::get_available_memory() {
	size_t bytes_available = 0;
	size_t bytes_total     = 0;
	CUDACALL(cuMemGetInfo(&bytes_available, &bytes_total));
	return bytes_available;
}

unsigned CUDAContext::get_shared_memory() { return device_get_attribute(CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK); }
unsigned CUDAContext::get_sm_count()      { return device_get_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT); }
