#include "CUDAModule.h"

#include <cstdio>
#include <cassert>
#include <vector>

#include <nvrtc.h>

#include "CUDAMemory.h"

#include "Util/Util.h"
#include "Util/ScopeTimer.h"

#define NVRTC_CALL(result) check_nvrtc_call(result, __FILE__, __LINE__);

static void check_nvrtc_call(nvrtcResult result, const char * file, int line) {
	if (result != NVRTC_SUCCESS) {
		const char * error_string = nvrtcGetErrorString(result);

		printf("NVRTC call at %s line %i failed with error %s!\n", file, line, error_string);
	  
		__debugbreak();
	}
}

struct Include {
	const char * filename;
	const char * source;
};

// Recursively walks include tree
// Collects filename and source of included files in 'includes'
// Also check whether any included file has been modified since last compilation and if so sets 'should_recompile'
// Returns source code of 'filename'
static const char * scan_includes_recursive(const char * filename, const char * directory, std::vector<Include> & includes, const char * ptx_filename, bool & should_recompile) {
	char * source = Util::file_read(filename);

	// Look for first #include of the file
	const char * include_ptr = strstr(source, "#include");

	while (include_ptr) {
		int include_start_index = include_ptr - source + 8;

		// Locate next < or " char
		const char * delimiter_lt_ptr = strchr(source + include_start_index, '<');
		const char * delimiter_qt_ptr = strchr(source + include_start_index, '\"');

		// Get the index of the next < and " chars, if they were found
		int delimiter_lt_index = delimiter_lt_ptr ? delimiter_lt_ptr - source : INT_MAX;
		int delimiter_qt_index = delimiter_qt_ptr ? delimiter_qt_ptr - source : INT_MAX;

		// Check whether < or " occurs first
		int include_filename_start_index = delimiter_lt_index < delimiter_qt_index ?
			delimiter_lt_index + 1 :
			delimiter_qt_index + 1;

		// Find the index of the next > or " char, depending on whether we previously saw a < or "
		int include_filename_end_index = delimiter_lt_index < delimiter_qt_index ?
			(strchr(source + include_filename_start_index, '>')  - source) :
			(strchr(source + include_filename_start_index, '\"') - source);

		// Allocate and copy over the filename of the include
		int    include_filename_length = include_filename_end_index - include_filename_start_index;
		char * include_filename = new char[include_filename_length + 1];

		memcpy_s(include_filename, include_filename_length, source + include_filename_start_index, include_filename_length);
		include_filename[include_filename_length] = NULL;
		
		// Check whether the include has been processed before
		bool unseen_include = true;

		for (const Include & include : includes) {
			if (strcmp(include.filename, include_filename) == 0) {
				unseen_include = false;

				break;
			}
		}

		// If we haven't seen this include before, recurse
		if (unseen_include) {
			int directory_length = strlen(directory);

			int    include_full_path_length = directory_length + include_filename_length + 1;
			char * include_full_path = MALLOCA(char, include_full_path_length);
			
			memcpy_s(include_full_path,                    include_full_path_length,                    directory,               directory_length);
			memcpy_s(include_full_path + directory_length, include_full_path_length - directory_length, include_filename, include_filename_length);
			include_full_path[include_full_path_length - 1] = NULL;

			if (Util::file_exists(include_full_path)) {
				if (!should_recompile && Util::file_is_newer(ptx_filename, include_full_path)) {
					should_recompile = true;

					printf("Recompilation required %s because included file %s changed.\n", filename, include_filename);
				}

				char * path = MALLOCA(char, include_full_path_length);
				Util::get_path(include_full_path, path);

				int index = includes.size();

				includes.emplace_back();
				includes[index].filename = include_filename;
				includes[index].source   = scan_includes_recursive(include_full_path, path, includes, ptx_filename, should_recompile);

				FREEA(path);
			}

			FREEA(include_full_path);
		}

		// Look for next #include, after the end of the current include
		include_ptr = strstr(source + include_filename_end_index, "#include");
	}

	return source;
}

void CUDAModule::init(const char * filename, int compute_capability, int max_registers) {
	ScopeTimer timer("CUDA Module Init");

	if (!Util::file_exists(filename)) {
		printf("ERROR: File %s does not exist!\n", filename);
		abort();
	}

	int    ptx_filename_size = strlen(filename) + 16;
	char * ptx_filename      = MALLOCA(char, ptx_filename_size);
#ifdef _DEBUG
	sprintf_s(ptx_filename, ptx_filename_size, "%s.debug.ptx",   filename);
#else
	sprintf_s(ptx_filename, ptx_filename_size, "%s.release.ptx", filename);
#endif
	
	bool should_recompile = true;

	// If the binary does not exists we definately need to compile
	if (Util::file_exists(ptx_filename)) {
		// Recompile if the source file is newer than the binary
		should_recompile = Util::file_is_newer(ptx_filename, filename);
	}

	char * path = MALLOCA(char, strlen(filename) + 1);
	Util::get_path(filename, path);

	std::vector<Include> includes;
	const char * source = scan_includes_recursive(filename, path, includes, ptx_filename, should_recompile);

	FREEA(path);

	if (should_recompile) {
		int num_includes = includes.size();

		const char ** include_names   = MALLOCA(const char *, num_includes);
		const char ** include_sources = MALLOCA(const char *, num_includes);
		
		for (int i = 0; i < num_includes; i++) {
			include_names  [i] = includes[i].filename;
			include_sources[i] = includes[i].source;
		}

		// Create NVRTC Program from the source and all includes
		nvrtcProgram program;
		NVRTC_CALL(nvrtcCreateProgram(&program, source, "Pathtracer", num_includes, include_sources, include_names));

		FREEA(include_names);
		FREEA(include_sources);

		// Configure options
		char compute    [64]; sprintf_s(compute,     "--gpu-architecture=compute_%i", compute_capability);
		char maxregcount[64]; sprintf_s(maxregcount, "--maxrregcount=%i", max_registers);

		const char * options[] = {
			"--std=c++17",
			compute,
			maxregcount,
			"--use_fast_math",
			"--extra-device-vectorization",
			//"--device-debug",
			"-lineinfo",
			"-restrict"
		};

		// Compile to PTX
		nvrtcResult result = nvrtcCompileProgram(program, Util::array_element_count(options), options);

		size_t log_size;
		NVRTC_CALL(nvrtcGetProgramLogSize(program, &log_size));

		if (log_size > 1) {
			char * log = new char[log_size];
			NVRTC_CALL(nvrtcGetProgramLog(program, log));

			puts("NVRTC output:");
			puts(log);

			delete [] log;
		}

		if (result != NVRTC_SUCCESS) abort();

		// Obtain PTX from NVRTC
		size_t ptx_size;                 NVRTC_CALL(nvrtcGetPTXSize(program, &ptx_size));
		char * ptx = new char[ptx_size]; NVRTC_CALL(nvrtcGetPTX    (program, ptx));

		NVRTC_CALL(nvrtcDestroyProgram(&program));

		// Cache PTX on disk
		FILE * file_out;
		fopen_s(&file_out, ptx_filename, "wb");

		fwrite(ptx, 1, ptx_size, file_out);
		fclose(file_out);
		
		delete [] ptx;
	} else {
		printf("CUDA Module %s did not need to recompile.\n", filename);
	}
	
	for (int i = 0; i < includes.size(); i++) {
		delete [] includes[i].filename;
		delete [] includes[i].source;
	}
	
	delete [] source;

	char log_buffer[8192];
	log_buffer[0] = NULL;

	CUjit_option options[] = {
		CU_JIT_MAX_REGISTERS,
		CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
		CU_JIT_INFO_LOG_BUFFER,
		CU_JIT_LOG_VERBOSE
	};
	void * values[] = {
		reinterpret_cast<void *>(max_registers),
		reinterpret_cast<void *>(sizeof(log_buffer)),
		reinterpret_cast<void *>(log_buffer),
		reinterpret_cast<void *>(should_recompile) // Only verbose if we just recompiled
	};

	CUlinkState link_state;
	CUDACALL(cuLinkCreate(Util::array_element_count(options), options, values, &link_state));
	CUDACALL(cuLinkAddFile(link_state, CU_JIT_INPUT_PTX, ptx_filename, 0, nullptr, nullptr));

	void * cubin;
	size_t cubin_size;
	cuLinkComplete(link_state, &cubin, &cubin_size);

	CUDACALL(cuModuleLoadData(&module, cubin));

	CUDACALL(cuLinkDestroy(link_state));
	
	if (should_recompile) puts(log_buffer);
	puts("");

	FREEA(ptx_filename);
}

void CUDAModule::free() {
	CUDACALL(cuModuleUnload(module));
}

CUDAModule::Global CUDAModule::get_global(const char * variable_name) const {
	Global global;

	size_t size;
	CUresult result = cuModuleGetGlobal(&global.ptr, &size, module, variable_name);
	if (result == CUDA_ERROR_NOT_FOUND) {
		printf("ERROR: Global CUDA variable '%s' not found!\n", variable_name);
	}
	CUDACALL(result);

	return global;
}
