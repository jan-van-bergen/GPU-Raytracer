#include "CUDAModule.h"

#include <cassert>
#include <cstdio>
#include <filesystem>

#include "CUDACall.h"

#include "ScopedTimer.h"

void CUDAModule::init(const char * filename, int compute_capability) {
	assert(std::filesystem::exists(filename));

	bool should_recompile = true;

	char output_filename[256];
	sprintf_s(output_filename, "%s.cubin", filename);

	// If the binary does not exists we definately need to compile
	if (std::filesystem::exists(output_filename)) {
		std::filesystem::file_time_type last_write_time_source = std::filesystem::last_write_time(       filename);
		std::filesystem::file_time_type last_write_time_cubin  = std::filesystem::last_write_time(output_filename);

		// Recompile if the source file is newer than the binary
		should_recompile = last_write_time_cubin < last_write_time_source;
	}

	if (should_recompile) {
		const int max_reg_count = 63;

		char command[256];
#ifdef _DEBUG
		sprintf_s(command, "nvcc -cubin -use_fast_math -I=\"lib\\CUDA\" -Xptxas=\"-v\" -lineinfo -maxrregcount=%i -arch=sm_%i -o %s %s 2>>%s 1>>%s", max_reg_count, compute_capability, output_filename, filename, "errorlog.txt", "buildlog.txt");
#else
		sprintf_s(command, "nvcc -cubin -use_fast_math -I=\"lib\\CUDA\" -restrict -Xptxas=\"-v\" -maxrregcount=%i -arch=sm_%i -o %s %s 2>>%s 1>>%s", max_reg_count, compute_capability, output_filename, filename, "errorlog.txt", "buildlog.txt" );
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
