#include "CUDAModule.h"

#include <cstdio>
#include <cassert>

#include <nvrtc.h>

#include "CUDAMemory.h"

#include "Util/Util.h"
#include "Util/Parser.h"
#include "Util/ScopeTimer.h"

#define NVRTC_CALL(result) check_nvrtc_call(result, __FILE__, __LINE__);

static void check_nvrtc_call(nvrtcResult result, const char * file, int line) {
	if (result != NVRTC_SUCCESS) {
		const char * error_string = nvrtcGetErrorString(result);

		printf("NVRTC call at %s line %i failed with error %s!\n", file, line, error_string);

		__debugbreak();
	}
}

// Recursively walks include tree
// Collects filename and source of included files in 'includes'
// Also check whether any included file has been modified since last compilation and if so sets 'should_recompile'
// Returns source code of 'filename'
static const char * scan_includes_recursive(const char * filename, StringView directory, Array<const char *> & include_names, Array<const char *> & include_sources, const char * ptx_filename, bool & should_recompile) {
	int    source_length;
	char * source = Util::file_read(filename, source_length);

	Parser parser = { };
	parser.init(source, source + source_length, filename);

	while (!parser.reached_end()) {
		if (parser.match("#include")) {
			parser.skip_whitespace();

			StringView include_filename = { };
			if (parser.match('<')) {
				include_filename.start = parser.cur;
				while (!parser.match('>')) {
					parser.advance();
				}
				include_filename.end = parser.cur - 1;
			} else if (parser.match('"')) {
				include_filename.start = parser.cur;
				while (!parser.match('"')) {
					parser.advance();
				}
				include_filename.end = parser.cur - 1;
			} else {
				ERROR(parser.location, "Invalid include token '%c', expected '\"' or '<'", *parser.cur);
			}

			// Check whether the include has been processed before
			bool unseen_include = true;

			for (int i = 0; i < include_names.size(); i++) {
				if (include_filename == include_names[i]) {
					unseen_include = false;

					break;
				}
			}

			// If we haven't seen this include before, recurse
			if (unseen_include) {
				const char * include_full_path = Util::get_absolute_path(directory, include_filename);

				if (Util::file_exists(include_full_path)) {
					if (!should_recompile && Util::file_is_newer(ptx_filename, include_full_path)) {
						should_recompile = true;

						printf("CUDA Module '%s': Recompilation required because included file '%.*s' changed.\n", filename, unsigned(include_filename.length()), include_filename.start);
					}

					StringView path = Util::get_directory(include_full_path);

					int index = include_names.size();
					include_names  .emplace_back();
					include_sources.emplace_back();

					include_names  [index] = include_filename.c_str();
					include_sources[index] = scan_includes_recursive(include_full_path, path, include_names, include_sources, ptx_filename, should_recompile);
				}

				delete [] include_full_path;
			}
		} else {
			parser.advance();
		}
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

	StringView path = Util::get_directory(filename);

	Array<const char *> include_names;
	Array<const char *> include_sources;
	const char * source = scan_includes_recursive(filename, path, include_names, include_sources, ptx_filename, should_recompile);
	assert(include_names.size() == include_sources.size());

	if (should_recompile) {
		nvrtcProgram program;

		while (true) {
			int num_includes = include_names.size();

			// Create NVRTC Program from the source and all includes
			NVRTC_CALL(nvrtcCreateProgram(&program, source, "Pathtracer", num_includes, include_sources.data(), include_names.data()));

			for (int i = 0; i < num_includes; i++) {
				delete [] include_names[i];
				delete [] include_sources[i];
			}
			include_names  .clear();
			include_sources.clear();

			delete [] source;

			// Configure options
			char compute    [64]; sprintf_s(compute,     "--gpu-architecture=compute_%i", compute_capability);
			char maxregcount[64]; sprintf_s(maxregcount, "--maxrregcount=%i", max_registers);

			const char * options[] = {
				"--std=c++11",
				compute,
				maxregcount,
				"--use_fast_math",
				"--extra-device-vectorization",
				//"--device-debug",
				"-lineinfo",
				"-restrict"
			};

			// Compile to PTX
			nvrtcResult result = nvrtcCompileProgram(program, Util::array_count(options), options);

			size_t log_size;
			NVRTC_CALL(nvrtcGetProgramLogSize(program, &log_size));

			if (log_size > 1) {
				char * log = new char[log_size];
				NVRTC_CALL(nvrtcGetProgramLog(program, log));

				puts("NVRTC output:");
				puts(log);

				delete [] log;
			}

			if (result == NVRTC_SUCCESS) break;

			__debugbreak(); // Compile error

			// Reload file and try again
			source = scan_includes_recursive(filename, path, include_names, include_sources, ptx_filename, should_recompile);
		}

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
	CUDACALL(cuLinkCreate(Util::array_count(options), options, values, &link_state));
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
