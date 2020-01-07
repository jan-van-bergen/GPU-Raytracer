#include "CUDAModule.h"

#include <cassert>
#include <cstdio>
#include <filesystem>

#include "CUDAMemory.h"

#include "ScopedTimer.h"

void CUDAModule::init(const char * filename, int compute_capability) {
	assert(std::filesystem::exists(filename));

	bool should_recompile = true;

	char output_filename[128];
#ifdef _DEBUG
	sprintf_s(output_filename, "%s.debug.cubin", filename);
#else
	sprintf_s(output_filename, "%s.release.cubin", filename);
#endif

	// If the binary does not exists we definately need to compile
	if (std::filesystem::exists(output_filename)) {
		std::filesystem::file_time_type last_write_time_source = std::filesystem::last_write_time(       filename);
		std::filesystem::file_time_type last_write_time_cubin  = std::filesystem::last_write_time(output_filename);

		// Recompile if the source file is newer than the binary
		should_recompile = last_write_time_cubin < last_write_time_source;
	}

	if (should_recompile) {
		char command[256];
#ifdef _DEBUG
		sprintf_s(command, "nvcc -cubin -use_fast_math -I=\"lib\\CUDA\" -Xptxas=\"-v\" -lineinfo -arch=sm_%i -o %s %s", compute_capability, output_filename, filename);
#else
		sprintf_s(command, "nvcc -cubin -use_fast_math -I=\"lib\\CUDA\" -restrict -Xptxas=\"-v\" -arch=sm_%i -o %s %s", compute_capability, output_filename, filename);
#endif
		printf("Compiling CUDA Module %s\n", filename);

		ScopedTimer timer("Compilation");

		int exit_code = system(command);
		if (exit_code != EXIT_SUCCESS) abort(); // Compilation failed!
	} else {
		printf("CUDA Module %s did not need to recompile.\n", filename);
	}

	CUDACALL(cuModuleLoad(&module, output_filename));
}

void CUDAModule::set_surface(const char * surface_name, CUarray array) const {
	CUsurfref surface;
	CUDACALL(cuModuleGetSurfRef(&surface, module, surface_name));

	CUDACALL(cuSurfRefSetArray(surface, array, 0));
}

void CUDAModule::set_texture(const char * texture_name, const Texture * texture) const {	
	// Create Array on Device
	CUarray tex_array = CUDAMemory::create_array(texture->width, texture->height, texture->channels, CU_AD_FORMAT_UNSIGNED_INT8);

	// Copy data from the Host Texture to the Device Array
	CUDAMemory::copy_array(tex_array, texture->channels * texture->width, texture->height, texture->data);

	// Set Texture parameters on Device
	CUtexref tex;
	CUDACALL(cuModuleGetTexRef(&tex, module, texture_name));
	CUDACALL(cuTexRefSetArray(tex, tex_array, CU_TRSA_OVERRIDE_FORMAT));

	CUDACALL(cuTexRefSetAddressMode(tex, 0, CU_TR_ADDRESS_MODE_WRAP));
	CUDACALL(cuTexRefSetAddressMode(tex, 1, CU_TR_ADDRESS_MODE_WRAP));

	CUDACALL(cuTexRefSetFilterMode(tex, CU_TR_FILTER_MODE_LINEAR));
	CUDACALL(cuTexRefSetFlags(tex, CU_TRSF_NORMALIZED_COORDINATES));
	CUDACALL(cuTexRefSetFormat(tex, CU_AD_FORMAT_UNSIGNED_INT8, texture->channels));
}

CUDAModule::Global CUDAModule::get_global(const char * variable_name) const {
	Global global;
	global.name = variable_name;

	size_t size;
	CUDACALL(cuModuleGetGlobal(&global.ptr, &size, module, global.name));

	return global;
}
