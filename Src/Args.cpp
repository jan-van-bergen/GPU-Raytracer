#include "Args.h"

#include "Config.h"

#include "Core/Parser.h"

#include "Math/Math.h"

static int parse_arg_int(StringView str) {
	return Parser(str).parse_int();
}

static float parse_arg_float(StringView str) {
	return Parser(str).parse_float();
}

static bool parse_arg_bool(StringView str) {
	if (
		str == "true" ||
		str == "True" ||
		str == "TRUE" ||
		str == "1") {
		return true;
	} else if (
		str == "false" ||
		str == "False" ||
		str == "FALSE" ||
		str == "0") {
		return false;
	} else {
		IO::print("Invalid boolean argument '{}'!\n"_sv, str);
		return true;
	}
};

static void parse_args(const Array<StringView> & args) {
	struct Option {
		StringView name_short;
		StringView name_full;

		StringView help_text;

		uint32_t num_args;

		void (* action)(const Array<StringView> & args, size_t i);
	};

	static Array<Option> options = {
		Option { "I"_sv, "integrator"_sv, "Choose the interagor type. Supported options: pathtracer, ao"_sv, 1, [](const Array<StringView> & args, size_t i) {
			if (args[i + 1] == "pathtracer") {
				cpu_config.integrator = IntegratorType::PATHTRACER;
			} else if (args[i + 1] == "ao") {
				cpu_config.integrator = IntegratorType::AO;
			} else {
				IO::print("'{}' is not a recognized integrator type! Supported options: pathtracer, ao\n"_sv, args[i + 1]);
				IO::exit(1);
			}
		} },

		Option { "W"_sv, "width"_sv,   "Sets the width of the window"_sv,                     1, [](const Array<StringView> & args, size_t i) { cpu_config.initial_width       = parse_arg_int(args[i + 1]); } },
		Option { "H"_sv, "height"_sv,  "Sets the height of the window"_sv,                    1, [](const Array<StringView> & args, size_t i) { cpu_config.initial_height      = parse_arg_int(args[i + 1]); } },
		Option { "b"_sv, "bounce"_sv,  "Sets the number of pathtracing bounces"_sv,           1, [](const Array<StringView> & args, size_t i) { gpu_config.num_bounces         = Math::clamp(parse_arg_int(args[i + 1]), 0, MAX_BOUNCES - 1); } },
		Option { "N"_sv, "samples"_sv, "Sets a target number of samples to use"_sv,           1, [](const Array<StringView> & args, size_t i) { cpu_config.output_sample_index = parse_arg_int(args[i + 1]); } },
		Option { "o"_sv, "output"_sv,  "Sets path to output file. Supported formats: ppm"_sv, 1, [](const Array<StringView> & args, size_t i) { cpu_config.output_filename     = args[i + 1].start; } },

		Option { "s"_sv, "scene"_sv, "Sets path to scene file. Supported formats: Mitsuba XML, OBJ, and PLY"_sv, 1, [](const Array<StringView> & args, size_t i) { cpu_config.scene_filenames.push_back(args[i + 1]); } },
		Option { "S"_sv, "sky"_sv,   "Sets path to sky file. Supported formats: HDR"_sv,                         1, [](const Array<StringView> & args, size_t i) { cpu_config.sky_filename = args[i + 1]; } },

		Option { "b"_sv, "bvh"_sv, "Sets type of BLAS BVH used. Supported options: sah, sbvh, bvh4, bvh8"_sv, 1, [](const Array<StringView> & args, size_t i) {
			if (args[i + 1] == "sah") {
				cpu_config.bvh_type = BVHType::BVH;
			} else if (args[i + 1] == "sbvh") {
				cpu_config.bvh_type = BVHType::SBVH;
			} else if (args[i + 1] == "bvh4") {
				cpu_config.bvh_type = BVHType::BVH4;
			} else if (args[i + 1] == "bvh8") {
				cpu_config.bvh_type = BVHType::BVH8;
			} else {
				IO::print("'{}' is not a recognized BVH type! Supported options: sah, sbvh, bvh4, bvh8\n"_sv, args[i + 1]);
				IO::exit(1);
			}
		} },

		Option { { }, "nee"_sv, "Enables or disables Next Event Estimation"_sv,        1, [](const Array<StringView> & args, size_t i) { gpu_config.enable_next_event_estimation        = parse_arg_bool(args[i + 1]); } },
		Option { { }, "mis"_sv, "Enables or disables Multiple Importance Sampling"_sv, 1, [](const Array<StringView> & args, size_t i) { gpu_config.enable_multiple_importance_sampling = parse_arg_bool(args[i + 1]); } },

		Option { { }, "force-rebuild"_sv, "BVH will not be loaded from disk but rebuild from scratch"_sv, 0, [](const Array<StringView> & args, size_t i) { cpu_config.bvh_force_rebuild = true; } },

		Option { "O"_sv,  "optimize"_sv,    "Enables or disables BVH optimzation post-processing step"_sv,               1, [](const Array<StringView> & args, size_t i) { cpu_config.enable_bvh_optimization       = parse_arg_bool(args[i + 1]); } },
		Option { "Ot"_sv, "opt-time"_sv,    "Sets time limit (in seconds) for BVH optimization"_sv,                      1, [](const Array<StringView> & args, size_t i) { cpu_config.bvh_optimizer_max_time        = parse_arg_int (args[i + 1]); } },
		Option { "Ob"_sv, "opt-batches"_sv, "Sets a limit on the maximum number of batches used in BVH optimization"_sv, 1, [](const Array<StringView> & args, size_t i) { cpu_config.bvh_optimizer_max_num_batches = parse_arg_int (args[i + 1]); } },

		Option { { }, "sah-node"_sv,   "Sets the SAH cost of an internal BVH node"_sv,                                                             1, [](const Array<StringView> & args, size_t i) { cpu_config.sah_cost_node = parse_arg_float(args[i + 1]); } },
		Option { { }, "sah-leaf"_sv,   "Sets the SAH cost of a leaf BVH node"_sv,                                                                  1, [](const Array<StringView> & args, size_t i) { cpu_config.sah_cost_leaf = parse_arg_float(args[i + 1]); } },
		Option { { }, "sbvh-alpha"_sv, "Sets the SBVH alpha constant. An alpha of 1 results in a regular BVH, alpha of 0 results in full SBVH"_sv, 1, [](const Array<StringView> & args, size_t i) { cpu_config.sbvh_alpha    = parse_arg_float(args[i + 1]); } },

		Option { { }, "mipmap"_sv,     "Enables or disables texture mipmapping"_sv,                                                     1, [](const Array<StringView> & args, size_t i) { gpu_config.enable_mipmapping = parse_arg_bool(args[i + 1]); } },
		Option { { }, "mip-filter"_sv, "Sets the downsampling filter for creating mipmaps: Supported options: box, lanczos, kaiser"_sv, 1, [](const Array<StringView> & args, size_t i) {
			if (args[i + 1] == "box") {
				cpu_config.mipmap_filter = MipmapFilterType::BOX;
			} else if (args[i + 1] == "lanczos") {
				cpu_config.mipmap_filter = MipmapFilterType::LANCZOS;
			} else if (args[i + 1] == "kaiser") {
				cpu_config.mipmap_filter = MipmapFilterType::KAISER;
			} else {
				IO::print("'{}' is not a recognized Mipmap Filter!\n"_sv, args[i + 1]);
				IO::exit(1);
			}
		} },
		Option { "c"_sv, "compress"_sv, "Enables or disables texture block compression"_sv, 1, [](const Array<StringView> & args, size_t i) { cpu_config.enable_block_compression = parse_arg_bool(args[i + 1]); } },
	};

	options.emplace_back("h"_sv, "help"_sv, "Displays this message"_sv, 0u, [](const Array<StringView> & args, size_t i) {
		for (int o = 0; o < options.size(); o++) {
			const Option & option = options[o];

			if (option.name_short.is_empty()) {
				IO::print("\t--{:16}{}\n"_sv, option.name_full, option.help_text);
			} else {
				IO::print("-{},\t--{:16}{}\n"_sv, option.name_short, option.name_full, option.help_text);
			}
		}
		exit(EXIT_SUCCESS);
	});

	for (size_t i = 1; i < args.size(); i++) {
		StringView arg = args[i];
		String arg_name = Format().format("Arg {} ({})"_sv, i, arg);

		Parser parser(arg, arg_name.view());

		if (parser.match('-')) {
			bool use_full_name = parser.match('-');

			bool match = false;

			for (int o = 0; o < options.size(); o++) {
				const Option & option = options[o];

				if (use_full_name) {
					match = parser.match(option.name_full);
				} else if (!option.name_short.is_empty()) {
					match = parser.match(option.name_short);
				}

				if (match && parser.reached_end()) {
					if (i + option.num_args >= args.size()) {
						IO::print("Not enough arguments provided to option '{}'!\n"_sv, option.name_full);
						return;
					}

					option.action(args, i);
					i += option.num_args;

					break;
				}
			}

			if (!match) {
				IO::print("Unrecognized command line option '{}'\nUse --help for a list of valid options\n"_sv, parser.cur);
			}
		} else {
			// Without explicit option, assume scene name
			cpu_config.scene_filenames.push_back(arg);
		}
	}
}

void Args::parse(int num_args, char ** args) {
	Array<StringView> arguments(num_args);
	for (int i = 0; i < num_args; i++) {
		arguments[i] = StringView::from_c_str(args[i]);
	}
	parse_args(arguments);
}
