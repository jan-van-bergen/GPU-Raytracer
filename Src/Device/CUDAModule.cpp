#include "CUDAModule.h"

#include <cstdio>
#include <cassert>

#include <nvrtc.h>

#include "CUDAMemory.h"

#include "Util/Util.h"
#include "Util/StringUtil.h"
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

struct Include {
	String filename;
	String source;
};

// Recursively walks include tree
// Collects filename and source of included files in 'includes'
// Also check whether any included file has been modified since last compilation and if so sets 'should_recompile'
// Returns source code of 'filename'
static String scan_includes_recursive(const String & filename, StringView directory, Array<Include> & includes, StringView ptx_filename, bool & should_recompile) {
	String source = Util::file_read(filename);

	Parser parser = { };
	parser.init(source.view(), filename.view());

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

			for (int i = 0; i < includes.size(); i++) {
				if (include_filename == includes[i].filename.view()) {
					unseen_include = false;

					break;
				}
			}

			// If we haven't seen this include before, recurse
			if (unseen_include) {
				String include_full_path = Util::combine_stringviews(directory, include_filename);

				if (Util::file_exists(include_full_path.view())) {
					if (!should_recompile && Util::file_is_newer(ptx_filename, include_full_path.view())) {
						should_recompile = true;

						printf("CUDA Module '%.*s': Recompilation required because included file '%.*s' changed.\n", FMT_STRING(filename), FMT_STRINGVIEW(include_filename));
					}

					StringView path = Util::get_directory(include_full_path.view());

					int index = includes.size();
					includes.emplace_back();

					includes[index].filename = String(include_filename);
					includes[index].source   = scan_includes_recursive(include_full_path, path, includes, ptx_filename, should_recompile);
				}
			}
		} else {
			parser.advance();
		}
	}

	return source;
}

void CUDAModule::init(const String & filename, int compute_capability, int max_registers) {
	ScopeTimer timer("CUDA Module Init");

	if (!Util::file_exists(filename.view())) {
		printf("ERROR: File %.*s does not exist!\n", FMT_STRING(filename));
		abort();
	}

#ifdef _DEBUG
	String ptx_filename = Util::combine_stringviews(filename.view(), StringView::from_c_str(".debug.ptx"));
#else
	String ptx_filename = Util::combine_stringviews(filename.view(), StringView::from_c_str(".release.ptx"));
#endif

	bool should_recompile = true;

	// If the binary does not exists we definately need to compile
	if (Util::file_exists(ptx_filename.view())) {
		// Recompile if the source file is newer than the binary
		should_recompile = Util::file_is_newer(ptx_filename.view(), filename.view());
	}

	StringView path = Util::get_directory(filename.view());

	Array<Include> includes;
	String source = scan_includes_recursive(filename, path, includes, ptx_filename.view(), should_recompile);

	if (should_recompile) {
		nvrtcProgram program;

		while (true) {
			int num_includes = includes.size();

			Array<const char *> include_names  (num_includes);
			Array<const char *> include_sources(num_includes);

			for (int i = 0; i < num_includes; i++) {
				include_names  [i] = includes[i].filename.data();
				include_sources[i] = includes[i].source  .data();
			}

			// Create NVRTC Program from the source and all includes
			NVRTC_CALL(nvrtcCreateProgram(&program, source.data(), "Pathtracer", num_includes, include_sources.data(), include_names.data()));

			includes.clear();

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
			source = scan_includes_recursive(filename, path, includes, ptx_filename.view(), should_recompile);
		}

		// Obtain PTX from NVRTC
		size_t ptx_size;           NVRTC_CALL(nvrtcGetPTXSize(program, &ptx_size));
		Array<char> ptx(ptx_size); NVRTC_CALL(nvrtcGetPTX    (program, ptx.data()));

		NVRTC_CALL(nvrtcDestroyProgram(&program));

		// Cache PTX on disk
		Util::file_write(ptx_filename, StringView { ptx.data(), ptx.data() + ptx.size() });
	} else {
		printf("CUDA Module '%.*s' did not need to recompile.\n", FMT_STRING(filename));
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
	CUDACALL(cuLinkAddFile(link_state, CU_JIT_INPUT_PTX, ptx_filename.data(), 0, nullptr, nullptr));

	void * cubin;
	size_t cubin_size;
	cuLinkComplete(link_state, &cubin, &cubin_size);

	CUDACALL(cuModuleLoadData(&module, cubin));

	CUDACALL(cuLinkDestroy(link_state));

	if (should_recompile) puts(log_buffer);
	puts("");
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
