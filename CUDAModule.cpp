#include "CUDAModule.h"

#include <cstdio>
#include <cassert>
#include <vector>
#include <string>
#include <fstream>

#include <nvrtc.h>

#include "CUDAMemory.h"

#include "Util.h"
#include "ScopedTimer.h"

static std::vector<std::string> get_includes(const std::string & filename) {
	std::vector<std::string> result;

	std::ifstream input(filename);

	std::string line;
	while (getline(input, line)) {
		if (line.rfind("#include \"", 0) == 0) {
			result.push_back(line.substr(10, line.length() - 11));
		}
	}

	return result;
}

void CUDAModule::init(const char * filename, int compute_capability, int max_registers) {
	assert(Util::file_exists(filename));

	bool should_recompile = true;

	char output_filename[128];
#ifdef _DEBUG
	sprintf_s(output_filename, "%s.debug.cubin", filename);
#else
	sprintf_s(output_filename, "%s.release.cubin", filename);
#endif

	// If the binary does not exists we definately need to compile
	if (Util::file_exists(output_filename)) {
		// Recompile if the source file is newer than the binary
		should_recompile = Util::file_is_newer(output_filename, filename);
	}
	
	// If any included file has changed we should recompile
	if (!should_recompile) {
		std::string directory = Util::get_path(filename);

		for (const std::string & include : get_includes(filename)) {
			// Recompile if the include file is newer than the binary
			if (Util::file_is_newer(output_filename, (directory + include).c_str())) {
				should_recompile = true;

				printf("Recompiling %s because included file %s changed.\n", filename, include.c_str());

				break;
			}
		}
	}

	if (should_recompile) {
		const char * ptx_args = "-v -warn-double-usage";

		char command[256];
#ifdef _DEBUG
		sprintf_s(command, "nvcc -cubin -use_fast_math -I=\"lib\\CUDA\" -Xptxas=\"%s\" -lineinfo -restrict -arch=sm_%i -maxrregcount=%i -o %s %s", ptx_args, compute_capability, max_registers, output_filename, filename);
#else
		sprintf_s(command, "nvcc -cubin -use_fast_math -I=\"lib\\CUDA\" -Xptxas=\"%s\" -lineinfo -restrict -arch=sm_%i -maxrregcount=%i -o %s %s", ptx_args, compute_capability, max_registers, output_filename, filename);
#endif
		printf("Compiling CUDA Module %s\n", filename);

		{
			ScopedTimer timer("Compilation");

			int exit_code = system(command);
			if (exit_code != EXIT_SUCCESS) abort(); // Compilation failed!
		}

#if false
		sprintf_s(command, "nvcc -ptx -use_fast_math -I=\"lib\\CUDA\" -Xptxas=\"-v\" -restrict -arch=sm_%i -maxrregcount=%i -o %s %s", compute_capability, max_registers, "CUDA_Source/debug.ptx", filename);
		
		{
			ScopedTimer timer("PTX");

			int exit_code = system(command);
			if (exit_code != EXIT_SUCCESS) abort(); // Compilation failed!
		}
#endif
	} else {
		printf("CUDA Module %s did not need to recompile.\n", filename);
	}

	puts("");

	CUDACALL(cuModuleLoad(&module, output_filename));
}

void CUDAModule::set_surface(const char * surface_name, CUarray array) const {
	CUsurfref surface;
	CUDACALL(cuModuleGetSurfRef(&surface, module, surface_name));

	CUDACALL(cuSurfRefSetArray(surface, array, 0));
}

void CUDAModule::set_texture(const char * texture_name, CUarray array, CUfilter_mode filter, CUarray_format format, int channels) const {
	CUtexref texture;
	CUDACALL(cuModuleGetTexRef(&texture, module, texture_name));
	CUDACALL(cuTexRefSetArray(texture, array, CU_TRSA_OVERRIDE_FORMAT));

	CUDACALL(cuTexRefSetAddressMode(texture, 0, CU_TR_ADDRESS_MODE_WRAP));
	CUDACALL(cuTexRefSetAddressMode(texture, 1, CU_TR_ADDRESS_MODE_WRAP));

	CUDACALL(cuTexRefSetFilterMode(texture, filter));
	CUDACALL(cuTexRefSetFlags(texture, CU_TRSF_NORMALIZED_COORDINATES));
	CUDACALL(cuTexRefSetFormat(texture, format, channels));
}

void CUDAModule::set_texture(const char * texture_name, const Texture * texture) const {	
	// Create Array on Device and copy Texture data over
	CUarray array = CUDAMemory::create_array(texture->width, texture->height, texture->channels, CU_AD_FORMAT_UNSIGNED_INT8);
	CUDAMemory::copy_array(array, texture->channels * texture->width, texture->height, texture->data);

	set_texture(texture_name, array, CU_TR_FILTER_MODE_LINEAR, CU_AD_FORMAT_UNSIGNED_INT8, texture->channels);
}

CUDAModule::Global CUDAModule::get_global(const char * variable_name) const {
	Global global;

	size_t size;
	CUDACALL(cuModuleGetGlobal(&global.ptr, &size, module, variable_name));

	return global;
}
