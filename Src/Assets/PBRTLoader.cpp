#include "PBRTLoader.h"

#include "PLYLoader.h"

#include "Util/Util.h"
#include "Util/Parser.h"
#include "Util/Geometry.h"

struct Instance {
	MeshDataHandle mesh_data_handle;
	MaterialHandle material_handle;
};

struct HomogeneousMedium {
	Vector3 sigma_a;
	Vector3 sigma_s;
};

using TextureMap  = HashMap<String, TextureHandle,     StringHash>;
using MaterialMap = HashMap<String, MaterialHandle,    StringHash>;
using MediumMap   = HashMap<String, HomogeneousMedium, StringHash>;
using ObjectMap   = HashMap<String, Array<Instance>,   StringHash>;

static void parser_next_line(Parser & parser) {
	while (true) {
		if (parser.reached_end()) return;

		if (is_newline(*parser.cur)) break;

		parser.advance();
	}
	parser.parse_newline();
}

// Skips whitespace, newlines, and comments
static void parser_skip(Parser & parser) {
	while (true) {
		if (parser.match('#')) { // Comment
			parser_next_line(parser);
		}
		if (!is_whitespace(*parser.cur) && !is_newline(*parser.cur)) break;

		parser.advance();
	}
}

static StringView parse_quoted(Parser & parser) {
	parser_skip(parser);
	parser.expect('"');

	const char * start = parser.cur;
	while (!parser.match('"')) {
		parser.advance();
	}

	return StringView { start, parser.cur - 1 };
}

struct Param {
	StringView name;

	enum struct Type {
		BOOL,
		INT,
		FLOAT,
		FLOAT2,
		FLOAT3,
		STRING,
		TEXTURE
	} type;

	Array<bool>       bools;
	Array<int>        ints;
	Array<float>      floats;
	Array<Vector2>    float2s;
	Array<Vector3>    float3s;
	Array<StringView> strings;

	SourceLocation location;
};

static Param parse_param(Parser & parser) {
	Param param = { };
	param.location = parser.location;

	StringView quoted = parse_quoted(parser);

	const char * space = quoted.start;
	while (*space != ' ') space++;

	StringView type = { quoted.start, space };
	StringView name = { space + 1, quoted.end };

	param.name = name;

	parser_skip(parser);
	bool has_brackets = parser.match('[');
	parser_skip(parser);

	if (type == "bool") {
		param.type = Param::Type::BOOL;
		do {
			bool value;
			if (parser.match("true")) {
				value = true;
			} else if (parser.match("false")) {
				value = false;
			} else {
				ERROR(parser.location, "Invalid boolean value!\n");
			}
			param.bools.push_back(value);
			parser_skip(parser);
		} while (has_brackets && !parser.match(']'));

	} else if (type == "int" || type == "integer") {
		param.type = Param::Type::INT;
		do {
			int value = parser.parse_int();
			param.ints.push_back(value);
			parser_skip(parser);
		} while (has_brackets && !parser.match(']'));

	} else if (type == "float" || type == "blackbody") {
		param.type = Param::Type::FLOAT;
		do {
			float value = parser.parse_float();
			param.floats.push_back(value);
			parser_skip(parser);
		} while (has_brackets && !parser.match(']'));

	} else if (type == "point2") {
		param.type = Param::Type::FLOAT2;
		do {
			Vector2 value;
			value.x = parser.parse_float(); parser_skip(parser);
			value.y = parser.parse_float();
			param.float2s.push_back(value);
			parser_skip(parser);
		} while (has_brackets && !parser.match(']'));

	} else if (type == "vector3" || type == "point3" || type == "normal" || type == "rgb") {
		param.type = Param::Type::FLOAT3;
		do {
			Vector3 value;
			value.x = parser.parse_float(); parser_skip(parser);
			value.y = parser.parse_float(); parser_skip(parser);
			value.z = parser.parse_float();
			param.float3s.push_back(value);
			parser_skip(parser);
		} while (has_brackets && !parser.match(']'));

	} else if (type == "string") {
		param.type = Param::Type::STRING;
		do {
			StringView value = parse_quoted(parser);
			param.strings.push_back(value);
			parser_skip(parser);
		} while (has_brackets && !parser.match(']'));

	} else if (type == "spectrum") {
		if (*parser.cur == '"') {
			param.type = Param::Type::STRING;
			do {
				StringView value = parse_quoted(parser);
				param.strings.push_back(value);
				parser_skip(parser);
			} while (has_brackets && !parser.match(']'));
		} else {
			param.type = Param::Type::FLOAT;
			do {
				float value = parser.parse_float();
				param.floats.push_back(value);
				parser_skip(parser);
			} while (has_brackets && !parser.match(']'));
		}

	} else if (type == "texture") {
		param.type = Param::Type::TEXTURE;
		do {
			StringView value = parse_quoted(parser);
			param.strings.push_back(value);
			parser_skip(parser);
		} while (has_brackets && !parser.match(']'));

	} else {
		ERROR(parser.location, "Unrecognized type '%.*s'", unsigned(type.length()), type.start);
	}

	parser_skip(parser);

	return param;
}

static Array<Param> parse_params(Parser & parser) {
	Array<Param> params;
	while (*parser.cur == '"') {
		params.push_back(parse_param(parser));
	}
	return params;
}

static const Param * find_param_optional(const Array<Param> & params, const char * name) {
	for (int i = 0; i < params.size(); i++) {
		if (params[i].name == name) {
			return &params[i];
		}
	}
	return nullptr;
}

static const Param * find_param_optional(const Array<Param> & params, const char * name, Param::Type type) {
	const Param * param = find_param_optional(params, name);
	if (param && param->type != type) {
		ERROR(param->location, "Param '%.*s' has invalid type!\n", unsigned(param->name.length()), param->name.start);
	}
	return param;
}

static const Param & find_param(const Array<Param> & params, const char * name) {
	const Param * param = find_param_optional(params, name);
	if (!param) {
		printf("Unable to find Param with name '%s'!\n", name);
	}
	return *param;
}

static const Param & find_param(const Array<Param> & params, const char * name, Param::Type type) {
	const Param & param = find_param(params, name);
	if (param.type != type) {
		ERROR(param.location, "Param '%.*s' has invalid type!\n", unsigned(param.name.length()), param.name.start);
	}
	return param;
}

static bool       find_param_bool   (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::BOOL)   .bools  [0]; }
static int        find_param_int    (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::INT)    .ints   [0]; }
static float      find_param_float  (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::FLOAT)  .floats [0]; }
static Vector2    find_param_float2 (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::FLOAT2) .float2s[0]; }
static Vector3    find_param_float3 (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::FLOAT3) .float3s[0]; }
static StringView find_param_string (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::STRING) .strings[0]; }
static StringView find_param_texture(const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::TEXTURE).strings[0]; }

static bool       find_param_bool   (const Array<Param> & params, const char * name, bool       fallback) { const Param * param = find_param_optional(params, name, Param::Type::BOOL);    return param ? param->bools  [0] : fallback; }
static int        find_param_int    (const Array<Param> & params, const char * name, int        fallback) { const Param * param = find_param_optional(params, name, Param::Type::INT);     return param ? param->ints   [0] : fallback; }
static float      find_param_float  (const Array<Param> & params, const char * name, float      fallback) { const Param * param = find_param_optional(params, name, Param::Type::FLOAT);   return param ? param->floats [0] : fallback; }
static Vector2    find_param_float2 (const Array<Param> & params, const char * name, Vector2    fallback) { const Param * param = find_param_optional(params, name, Param::Type::FLOAT2);  return param ? param->float2s[0] : fallback; }
static Vector3    find_param_float3 (const Array<Param> & params, const char * name, Vector3    fallback) { const Param * param = find_param_optional(params, name, Param::Type::FLOAT3);  return param ? param->float3s[0] : fallback; }
static StringView find_param_string (const Array<Param> & params, const char * name, StringView fallback) { const Param * param = find_param_optional(params, name, Param::Type::STRING);  return param ? param->strings[0] : fallback; }
static StringView find_param_texture(const Array<Param> & params, const char * name, StringView fallback) { const Param * param = find_param_optional(params, name, Param::Type::TEXTURE); return param ? param->strings[0] : fallback; }

static const Array<bool>       & find_param_bools   (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::BOOL)   .bools; }
static const Array<int>        & find_param_ints    (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::INT)    .ints; }
static const Array<float>      & find_param_floats  (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::FLOAT)  .floats; }
static const Array<Vector2>    & find_param_float2s (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::FLOAT2) .float2s; }
static const Array<Vector3>    & find_param_float3s (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::FLOAT3) .float3s; }
static const Array<StringView> & find_param_strings (const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::STRING) .strings; }
static const Array<StringView> & find_param_textures(const Array<Param> & params, const char * name) { return find_param(params, name, Param::Type::TEXTURE).strings; }

static const char * get_absolute_filename(const char * path, int len_path, const char * filename, int len_filename) {
	char * filename_abs = new char[len_path + len_filename + 1];

	memcpy(filename_abs,            path,     len_path);
	memcpy(filename_abs + len_path, filename, len_filename);
	filename_abs[len_path + len_filename] = '\0';

	return filename_abs;
}

static Material parse_material(const char * name, const StringView & type, const Array<Param> & params, const TextureMap & texture_map) {
	Material material = { };
	material.name = name;

	auto parse_reflectance = [&material, &params, &texture_map]() {
		const Param * param_reflectance = find_param_optional(params, "reflectance");
		if (!param_reflectance) return;

		if (param_reflectance->type == Param::Type::FLOAT3) {
			material.diffuse = param_reflectance->float3s[0];
		} else if (param_reflectance->type == Param::Type::TEXTURE) {
			StringView texture_name = param_reflectance->strings[0];

			if (!texture_map.try_get(texture_name, material.texture_id)) {
				WARNING(param_reflectance->location, "Undefined texture '%.*s'!\n", unsigned(texture_name.length()), texture_name.start);
			}
		} else {
			ERROR(param_reflectance->location, "Invalid type!\n");
		}
	};

	if (type == "diffuse" || type == "diffusetransmission" || type == "subsurface") {
		material.type = Material::Type::DIFFUSE;
		parse_reflectance();
	} else if (type == "coateddiffuse") {
		material.type = Material::Type::GLOSSY;
		parse_reflectance();

		const Param * param_roughness = find_param_optional(params, "roughness");
		if (param_roughness && param_roughness->type == Param::Type::FLOAT) {
			material.linear_roughness = sqrtf(param_roughness->floats[0]);
		} else {
			material.linear_roughness = 0.5f;
		}
	} else if (type == "conductor") {
		material.type = Material::Type::GLOSSY;

		const Param * param_eta = find_param_optional(params, "eta");
		const Param * param_k   = find_param_optional(params, "k");

		struct IOR {
			const char * name;
			Vector3      eta;
			Vector3      k;
		};

		// Based on: https://github.com/tunabrain/tungsten/blob/master/src/core/bsdfs/ComplexIorData.hpp
		static IOR known_iors[] = {
			{ "a-C",    Vector3( 2.9440999183f, 2.2271502925f, 1.9681668794f), Vector3( 0.8874329109f,  0.7993216383f, 0.8152862927f) },
			{ "Ag",     Vector3( 0.1552646489f, 0.1167232965f, 0.1383806959f), Vector3( 4.8283433224f,  3.1222459278f, 2.1469504455f) },
			{ "Al",     Vector3( 1.6574599595f, 0.8803689579f, 0.5212287346f), Vector3( 9.2238691996f,  6.2695232477f, 4.8370012281f) },
			{ "AlAs",   Vector3( 3.6051023902f, 3.2329365777f, 2.2175611545f), Vector3( 0.0006670247f, -0.0004999400f, 0.0074261204f) },
			{ "AlSb",   Vector3(-0.0485225705f, 4.1427547893f, 4.6697691348f), Vector3(-0.0363741915f,  0.0937665154f, 1.3007390124f) },
			{ "Au",     Vector3( 0.1431189557f, 0.3749570432f, 1.4424785571f), Vector3( 3.9831604247f,  2.3857207478f, 1.6032152899f) },
			{ "Be",     Vector3( 4.1850592788f, 3.1850604423f, 2.7840913457f), Vector3( 3.8354398268f,  3.0101260162f, 2.8690088743f) },
			{ "Cr",     Vector3( 4.3696828663f, 2.9167024892f, 1.6547005413f), Vector3( 5.2064337956f,  4.2313645277f, 3.7549467933f) },
			{ "CsI",    Vector3( 2.1449030413f, 1.7023164587f, 1.6624194173f), Vector3( 0.0000000000f,  0.0000000000f, 0.0000000000f) },
			{ "Cu",     Vector3( 0.2004376970f, 0.9240334304f, 1.1022119527f), Vector3( 3.9129485033f,  2.4528477015f, 2.1421879552f) },
			{ "Cu2O",   Vector3( 3.5492833755f, 2.9520622449f, 2.7369202137f), Vector3( 0.1132179294f,  0.1946659670f, 0.6001681264f) },
			{ "CuO",    Vector3( 3.2453822204f, 2.4496293965f, 2.1974114493f), Vector3( 0.5202739621f,  0.5707372756f, 0.7172250613f) },
			{ "d-C",    Vector3( 2.7112524747f, 2.3185812849f, 2.2288565009f), Vector3( 0.0000000000f,  0.0000000000f, 0.0000000000f) },
			{ "Hg",     Vector3( 2.3989314904f, 1.4400254917f, 0.9095512090f), Vector3( 6.3276269444f,  4.3719414152f, 3.4217899270f) },
			{ "HgTe",   Vector3( 4.7795267752f, 3.2309984581f, 2.6600252401f), Vector3( 1.6319827058f,  1.5808189339f, 1.7295753852f) },
			{ "Ir",     Vector3( 3.0864098394f, 2.0821938440f, 1.6178866805f), Vector3( 5.5921510077f,  4.0671757150f, 3.2672611269f) },
			{ "K",      Vector3( 0.0640493070f, 0.0464100621f, 0.0381842017f), Vector3( 2.1042155920f,  1.3489364357f, 0.9132113889f) },
			{ "Li",     Vector3( 0.2657871942f, 0.1956102432f, 0.2209198538f), Vector3( 3.5401743407f,  2.3111306542f, 1.6685930000f) },
			{ "MgO",    Vector3( 2.0895885542f, 1.6507224525f, 1.5948759692f), Vector3( 0.0000000000f, -0.0000000000f, 0.0000000000f) },
			{ "Mo",     Vector3( 4.4837010280f, 3.5254578255f, 2.7760769438f), Vector3( 4.1111307988f,  3.4208716252f, 3.1506031404f) },
			{ "Na",     Vector3( 0.0602665320f, 0.0561412435f, 0.0619909494f), Vector3( 3.1792906496f,  2.1124800781f, 1.5790940266f) },
			{ "Nb",     Vector3( 3.4201353595f, 2.7901921379f, 2.3955856658f), Vector3( 3.4413817900f,  2.7376437930f, 2.5799132708f) },
			{ "Ni",     Vector3( 2.3672753521f, 1.6633583302f, 1.4670554172f), Vector3( 4.4988329911f,  3.0501643957f, 2.3454274399f) },
			{ "Rh",     Vector3( 2.5857954933f, 1.8601866068f, 1.5544279524f), Vector3( 6.7822927110f,  4.7029501026f, 3.9760892461f) },
			{ "Se-e",   Vector3( 5.7242724833f, 4.1653992967f, 4.0816099264f), Vector3( 0.8713747439f,  1.1052845009f, 1.5647788766f) },
			{ "Se",     Vector3( 4.0592611085f, 2.8426947380f, 2.8207582835f), Vector3( 0.7543791750f,  0.6385150558f, 0.5215872029f) },
			{ "SiC",    Vector3( 3.1723450205f, 2.5259677964f, 2.4793623897f), Vector3( 0.0000007284f, -0.0000006859f, 0.0000100150f) },
			{ "SnTe",   Vector3( 4.5251865890f, 1.9811525984f, 1.2816819226f), Vector3( 0.0000000000f,  0.0000000000f, 0.0000000000f) },
			{ "Ta",     Vector3( 2.0625846607f, 2.3930915569f, 2.6280684948f), Vector3( 2.4080467973f,  1.7413705864f, 1.9470377016f) },
			{ "Te-e",   Vector3( 7.5090397678f, 4.2964603080f, 2.3698732430f), Vector3( 5.5842076830f,  4.9476231084f, 3.9975145063f) },
			{ "Te",     Vector3( 7.3908396088f, 4.4821028985f, 2.6370708478f), Vector3( 3.2561412892f,  3.5273908133f, 3.2921683116f) },
			{ "ThF4",   Vector3( 1.8307187117f, 1.4422274283f, 1.3876488528f), Vector3( 0.0000000000f,  0.0000000000f, 0.0000000000f) },
			{ "TiC",    Vector3( 3.7004673762f, 2.8374356509f, 2.5823030278f), Vector3( 3.2656905818f,  2.3515586388f, 2.1727857800f) },
			{ "TiN",    Vector3( 1.6484691607f, 1.1504482522f, 1.3797795097f), Vector3( 3.3684596226f,  1.9434888540f, 1.1020123347f) },
			{ "TiO2-e", Vector3( 3.1065574823f, 2.5131551146f, 2.5823844157f), Vector3( 0.0000289537f, -0.0000251484f, 0.0001775555f) },
			{ "TiO2",   Vector3( 3.4566203131f, 2.8017076558f, 2.9051485020f), Vector3( 0.0001026662f, -0.0000897534f, 0.0006356902f) },
			{ "VC",     Vector3( 3.6575665991f, 2.7527298065f, 2.5326814570f), Vector3( 3.0683516659f,  2.1986687713f, 1.9631816252f) },
			{ "VN",     Vector3( 2.8656011588f, 2.1191817791f, 1.9400767149f), Vector3( 3.0323264950f,  2.0561075580f, 1.6162930914f) },
			{ "V",      Vector3( 4.2775126218f, 3.5131538236f, 2.7611257461f), Vector3( 3.4911844504f,  2.8893580874f, 3.1116965117f) },
			{ "W",      Vector3( 4.3707029924f, 3.3002972445f, 2.9982666528f), Vector3( 3.5006778591f,  2.6048652781f, 2.2731930614f) }
		};

		if (param_eta) {
			if (param_eta->type == Param::Type::FLOAT) {
				material.eta = Vector3(param_eta->floats[0]);
			} else if (param_eta->type == Param::Type::FLOAT3) {
				material.eta = param_eta->float3s[0];
			} else if (param_eta->type == Param::Type::STRING) {
				StringView metal_name_full = param_eta->strings[0];
				StringView metal_name      = { };

				Parser parser = { };
				parser.init(metal_name_full.start, metal_name_full.end);

				parser.expect("metal-");
				metal_name.start = parser.cur;
				while (*parser.cur != '-') {
					parser.advance();
				}
				metal_name.end = parser.cur;
				parser.expect("-eta");

				bool found = false;

				for (int i = 0; i < Util::array_count(known_iors); i++) {
					if (metal_name == known_iors[i].name) {
						material.eta = known_iors[i].eta;
						found = true;
						break;
					}
				}

				if (!found) {
					WARNING(param_eta->location, "Unknown metal eta '%.*s'!\n", unsigned(metal_name_full.length()), metal_name_full.start);
				}
			}
		}

		if (param_k) {
			if (param_k->type == Param::Type::FLOAT) {
				material.k = Vector3(param_k->floats[0]);
			} else if (param_k->type == Param::Type::FLOAT3) {
				material.k = param_k->float3s[0];
			} else if (param_k->type == Param::Type::STRING) {
				StringView metal_name_full = param_k->strings[0];
				StringView metal_name      = { };

				Parser parser = { };
				parser.init(metal_name_full.start, metal_name_full.end);

				parser.expect("metal-");
				metal_name.start = parser.cur;
				while (*parser.cur != '-') {
					parser.advance();
				}
				metal_name.end = parser.cur;
				parser.expect("-k");

				bool found = false;

				for (int i = 0; i < Util::array_count(known_iors); i++) {
					if (metal_name == known_iors[i].name) {
						material.k = known_iors[i].k;
						found = true;
						break;
					}
				}

				if (!found) {
					WARNING(param_k->location, "Unknown metal k '%.*s'!\n", unsigned(metal_name_full.length()), metal_name_full.start);
				}
			}
		}

	} else if (type == "dielectric" || type == "thindielectric") {
		material.type = Material::Type::DIELECTRIC;
		material.transmittance       = Vector3(1.0f);
		material.index_of_refraction = 1.333f;
	} else {
		printf("WARNING: Material Type '%.*s' is not supported!\n", unsigned(type.length()), type.start);
	}

	return material;
}

static void load_include(const char * filename, const char * path, int path_length, Scene & scene, TextureMap & texture_map, MaterialMap & material_map, MediumMap & medium_map, ObjectMap & object_map) {
	int          file_length;
	const char * file = Util::file_read(filename, file_length);

	SourceLocation location;
	location.file = filename;
	location.line = 1;
	location.col  = 0;

	Parser parser;
	parser.init(file, file + file_length, location);

	struct Attribute {
		MaterialHandle material = MaterialHandle::get_default();
		Matrix4        transform;
	};

	Array<Attribute> attribute_stack;
	attribute_stack.emplace_back();

	struct {
		bool inside = false;

		String          name;
		Array<Instance> shapes;
	} current_object;

	while (!parser.reached_end()) {
		parser.skip_whitespace();

		if (parser.match('#')) {
			parser_next_line(parser);

		} else if (parser.match("Include")) {
			StringView name = parse_quoted(parser);
			parser_skip(parser);

			const char * include_abs = get_absolute_filename(path, path_length, name.start, name.length());
			load_include(include_abs, path, path_length, scene, texture_map, material_map, medium_map, object_map);
			delete [] include_abs;

		} else if (parser.match("AttributeBegin")) {
			parser_skip(parser);
			if (attribute_stack.size() == 0) {
				ERROR(parser.location, "Invalid AttributeBegin block!\n");
			}
			attribute_stack.emplace_back(attribute_stack.back());

		} else if (parser.match("AttributeEnd")) {
			parser_skip(parser);

			if (attribute_stack.size() == 0) {
				ERROR(parser.location, "Invalid AttributeEnd block!\n");
			}
			attribute_stack.pop_back();

		} else if (parser.match("ObjectBegin")) {
			StringView name = parse_quoted(parser);
			parser_skip(parser);

			if (current_object.inside) {
				ERROR(parser.location, "Nested Objects!\n");
			}
			current_object.inside = true;
			current_object.name   = name;
			current_object.shapes.clear();

		} else if (parser.match("ObjectEnd")) {
			parser_skip(parser);

			if (!current_object.inside) {
				ERROR(parser.location, "ObjectEnd without ObjectBegin!\n");
			}
			current_object.inside = false;

			object_map.insert(current_object.name, std::move(current_object.shapes));

		} else if (parser.match("ObjectInstance")) {
			StringView name = parse_quoted(parser);
			parser_skip(parser);

			Array<Instance> shape;
			if (!object_map.try_get(name, shape)) {
				ERROR(parser.location, "Trying to create instance of unknown shape '%.*s'!\n", unsigned(name.length()), name.start);
			}

			for (int i = 0; i < shape.size(); i++) {
				const Instance & instance = shape[i];

				Mesh & mesh = scene.add_mesh(name.c_str(), instance.mesh_data_handle, instance.material_handle);
				Matrix4::decompose(attribute_stack.back().transform, &mesh.position, &mesh.rotation, &mesh.scale);
			}

		} else if (parser.match("Identity")) {
			parser_skip(parser);

			attribute_stack.back().transform = Matrix4();

		} else if (parser.match("Translate")) {
			parser_skip(parser);
			float x = parser.parse_float(); parser_skip(parser);
			float y = parser.parse_float(); parser_skip(parser);
			float z = parser.parse_float(); parser_skip(parser);

			attribute_stack.back().transform = Matrix4::create_translation(Vector3(x, y, z)) * attribute_stack.back().transform;

		} else if (parser.match("Rotate")) {
			parser_skip(parser);
			float angle = Math::deg_to_rad(parser.parse_float()); parser_skip(parser);
			float x = parser.parse_float(); parser_skip(parser);
			float y = parser.parse_float(); parser_skip(parser);
			float z = parser.parse_float(); parser_skip(parser);

			attribute_stack.back().transform = Matrix4::create_rotation(Quaternion::axis_angle(Vector3(x, y, z), angle)) * attribute_stack.back().transform;

		} else if (parser.match("Scale")) {
			parser_skip(parser);
			float x = parser.parse_float(); parser_skip(parser);
			float y = parser.parse_float(); parser_skip(parser);
			float z = parser.parse_float(); parser_skip(parser);

			attribute_stack.back().transform = Matrix4::create_scale(cbrtf(x * y * z)) * attribute_stack.back().transform; // Geometric mean

		} else if (parser.match("Transform")) {
			parser_skip(parser);
			parser.expect('[');
			parser_skip(parser);

			Matrix4 matrix;
			for (int i = 0; i < 16; i++) {
				matrix.cells[i] = parser.parse_float();
				parser_skip(parser);
			}
//			attribute_stack.back().transform = Matrix4::transpose(matrix);
			attribute_stack.back().transform = matrix;

			parser.expect(']');
			parser_skip(parser);

		} else if (parser.match("ConcatTransform")) {
			parser_skip(parser);
			parser.expect('[');
			parser_skip(parser);

			Matrix4 matrix;
			for (int i = 0; i < 16; i++) {
				matrix.cells[i] = parser.parse_float();
				parser_skip(parser);
			}
//			attribute_stack.back().transform = Matrix4::transpose(matrix) * attribute_stack.back().transform;
			attribute_stack.back().transform = matrix * attribute_stack.back().transform;

			parser.expect(']');
			parser_skip(parser);

		} else if (parser.match("LookAt")) {
			parser_skip(parser);
			float e_x = parser.parse_float(); parser_skip(parser);
			float e_y = parser.parse_float(); parser_skip(parser);
			float e_z = parser.parse_float(); parser_skip(parser);
			float f_x = parser.parse_float(); parser_skip(parser);
			float f_y = parser.parse_float(); parser_skip(parser);
			float f_z = parser.parse_float(); parser_skip(parser);
			float u_x = parser.parse_float(); parser_skip(parser);
			float u_y = parser.parse_float(); parser_skip(parser);
			float u_z = parser.parse_float(); parser_skip(parser);

			attribute_stack.back().transform =
				Matrix4::create_translation(Vector3(e_x, e_y, e_z)) *
				Matrix4::create_rotation(Quaternion::look_rotation(Vector3(f_x, f_y, f_z), Vector3(u_x, u_y, u_z)));

		} else if (parser.match("Camera")) {
			StringView type = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);

			if (type == "perspective") {
				scene.camera.set_fov(Math::deg_to_rad(find_param_float(params, "fov", 90.0f)));
				scene.camera.aperture_radius = find_param_float(params, "lensradius", 0.1f);
				scene.camera.focal_distance  = find_param_float(params, "focaldistance", 10.0f);
			} else {
				WARNING(parser.location, "Unsupported Camera type '%.*s'!\n", unsigned(type.length()), type.start);
			}

			Matrix4::decompose(attribute_stack.back().transform, &scene.camera.position, &scene.camera.rotation, nullptr);
		} else if (parser.match("Texture")) {
			StringView name = parse_quoted(parser);
			StringView type = parse_quoted(parser);
			StringView clas = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);

			if (clas == "imagemap") {
				StringView   filename_rel = find_param_string(params, "filename");
				const char * filename_abs = get_absolute_filename(path, path_length, filename_rel.start, filename_rel.length());

				TextureHandle texture_handle = scene.asset_manager.add_texture(filename_abs);
				texture_map.insert(name, texture_handle);

				delete [] filename_abs;

			} else if (clas == "scale") {
				StringView tex = find_param_texture(params, "tex");

				TextureHandle texture_handle;
				if (texture_map.try_get(tex, texture_handle)) {
					texture_map.insert(name, texture_handle); // Reinsert under new name
				}
			} else if (clas == "mix") {
				StringView texture_name = { };

				const Param & param_tex1   = find_param(params, "tex1");
				const Param & param_tex2   = find_param(params, "tex2");
				const Param & param_amount = find_param(params, "amount");

				if (param_tex1.type == Param::Type::TEXTURE) {
					texture_name = param_tex1.strings[0];
				} else if (param_tex2.type == Param::Type::TEXTURE) {
					texture_name = param_tex2.strings[0];
				} else if (param_tex2.type == Param::Type::TEXTURE) {
					texture_name = param_amount.strings[0];
				}

				TextureHandle texture_handle;
				if (texture_map.try_get(texture_name, texture_handle)) {
					texture_map.insert(name, texture_handle); // Reinsert under new name
				}
			} else {
				WARNING(parser.location, "Unsupported texture class '%.*s'!\n", unsigned(clas.length()), clas.start);
			}

		} else if (parser.match("Material")) {
			StringView type = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);

			Material material = parse_material("Material", type, params, texture_map);
			attribute_stack.back().material = scene.asset_manager.add_material(material);

		} else if (parser.match("MakeNamedMaterial")) {
			StringView name = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);
			StringView type = find_param_string(params, "type");

			Material       material        = parse_material(name.c_str(), type, params, texture_map);
			MaterialHandle material_handle = scene.asset_manager.add_material(material);

			material_map.insert(name, material_handle);

		} else if (parser.match("NamedMaterial")) {
			StringView name = parse_quoted(parser);
			parser_skip(parser);

			if (!material_map.try_get(name, attribute_stack.back().material)) {
				attribute_stack.back().material = MaterialHandle::get_default();
				WARNING(parser.location, "Used undefined NamedMaterial '%.*s'!\n", unsigned(name.length()), name.start)
			}

		} else if (parser.match("MakeNamedMedium")) {
			StringView name = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);
			StringView type = find_param_string(params, "type");

			if (type == "homogeneous") {
				const Param * param_sigma_a = find_param_optional(params, "sigma_a");
				const Param * param_sigma_s = find_param_optional(params, "sigma_s");

				HomogeneousMedium medium = { };
				if (param_sigma_a && param_sigma_a->type == Param::Type::FLOAT3) {
					medium.sigma_a = param_sigma_a->float3s[0];
				} else {
					medium.sigma_a = Vector3(0.0011f, 0.0024f, 0.014f);
				}
				if (param_sigma_s && param_sigma_s->type == Param::Type::FLOAT3) {
					medium.sigma_s = param_sigma_s->float3s[0];
				} else {
					medium.sigma_s = Vector3(2.55f, 3.21f, 3.77f);
				}

				medium_map.insert(name, medium);
			} else {
				WARNING(parser.location, "Only homogeneous media are supported!\n");
			}

		} else if (parser.match("MediumInterface")) {
			StringView medium_name_from = parse_quoted(parser); parser_skip(parser);
			StringView medium_name_to   = parse_quoted(parser); parser_skip(parser);

			HomogeneousMedium medium;
			if (medium_map.try_get(medium_name_to, medium)) {
				MaterialHandle material_handle = attribute_stack.back().material;
				if (material_handle.handle != INVALID) {
					Material & material = scene.asset_manager.get_material(material_handle);
					if (material.type == Material::Type::DIELECTRIC) {
						material.transmittance = Vector3(
							expf(-(medium.sigma_a.x + medium.sigma_s.x)),
							expf(-(medium.sigma_a.y + medium.sigma_s.y)),
							expf(-(medium.sigma_a.z + medium.sigma_s.z))
						);
					}
				}
			} else {
				WARNING(parser.location, "Named Medium '%.*s' not found!\n", unsigned(medium_name_to.length()), medium_name_to.start);
			}

		} else if (parser.match("AreaLightSource")) {
			StringView type = parse_quoted(parser);
			parser_skip(parser);

			if (type != "diffuse") {
				WARNING(parser.location, "AreaLightSource type should be 'diffuse'!\n");
			}

			Array<Param> params = parse_params(parser);

			Material material = { };
			material.name = "AreaLightSource";
			material.type = Material::Type::LIGHT;

			const Param * emit = find_param_optional(params, "L");
			if (emit) {
				if (emit->type == Param::Type::FLOAT3) {
					material.emission = emit->float3s[0];
				} else if (emit->type == Param::Type::FLOAT) {
					material.emission = Vector3(1.0f); ///////////////////////////////////////////////////////////////////////////////////////////////////
				}
			} else {
				material.emission = Vector3(1.0f);
			}

			const Param * scale = find_param_optional(params, "scale");
			if (scale) {
				material.emission *= scale->floats[0];
			}

			attribute_stack.back().material = scene.asset_manager.add_material(material);

		} else if (parser.match("Shape")) {
			StringView type = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);

			if (type == "plymesh") {
				StringView param_filename = find_param(params, "filename", Param::Type::STRING).strings[0];

				const char * filename_rel = param_filename.c_str();
				const char * filename_abs = get_absolute_filename(path, path_length, filename_rel, param_filename.length());

				MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(filename_abs, PLYLoader::load);

				if (current_object.inside) {
					Instance instance = { };
					instance.mesh_data_handle = mesh_data_handle;
					instance.material_handle  = attribute_stack.back().material;
					current_object.shapes.push_back(instance);
				} else {
					Mesh & mesh = scene.add_mesh(filename_rel, mesh_data_handle, attribute_stack.back().material);
					Matrix4::decompose(attribute_stack.back().transform, &mesh.position, &mesh.rotation, &mesh.scale);
				}

				delete [] filename_abs;
			} else if (type == "trianglemesh") {
				const Param & positions  = find_param         (params, "P",  Param::Type::FLOAT3);
				const Param * normals    = find_param_optional(params, "N",  Param::Type::FLOAT3);
				const Param * tex_coords = find_param_optional(params, "uv", Param::Type::FLOAT2);

				const Param & param_indices = find_param(params, "indices", Param::Type::INT);
				const Array<int> & indices = param_indices.ints;

				if (indices.size() % 3 != 0) {
					WARNING(param_indices.location, "The number of Triangle Mesh indices should be a multiple of 3!\n");
				}

				int        triangle_count = indices.size() / 3;
				Triangle * triangles      = new Triangle[triangle_count];

				for (int i = 0; i < triangle_count; i++) {
					int index_0 = indices[3*i];
					int index_1 = indices[3*i + 1];
					int index_2 = indices[3*i + 2];

					Triangle & triangle = triangles[i];

					triangle.position_0 = positions.float3s[index_0];
					triangle.position_1 = positions.float3s[index_1];
					triangle.position_2 = positions.float3s[index_2];

					if (normals) {
						triangle.normal_0 = normals->float3s[index_0];
						triangle.normal_1 = normals->float3s[index_1];
						triangle.normal_2 = normals->float3s[index_2];
					} else {
						Vector3 geometric_normal = Vector3::normalize(Vector3::cross(
							triangle.position_1 - triangle.position_0,
							triangle.position_2 - triangle.position_0
						));
						triangle.normal_0 = geometric_normal;
						triangle.normal_1 = geometric_normal;
						triangle.normal_2 = geometric_normal;
					}

					if (tex_coords) {
						triangle.tex_coord_0 = tex_coords->float2s[index_0];
						triangle.tex_coord_1 = tex_coords->float2s[index_1];
						triangle.tex_coord_2 = tex_coords->float2s[index_2];
					} else {
						triangle.tex_coord_0 = Vector2(0.0f);
						triangle.tex_coord_1 = Vector2(0.0f);
						triangle.tex_coord_2 = Vector2(0.0f);
					}

					triangle.init();
				}

				MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(triangles, triangle_count);
				Mesh & mesh = scene.add_mesh("Triangle Mesh", mesh_data_handle, attribute_stack.back().material);
				Matrix4::decompose(attribute_stack.back().transform, &mesh.position, &mesh.rotation, &mesh.scale);
			} else if (type == "sphere") {
				float radius = find_param_float(params, "radius", 1.0f);

				int        triangle_count;
				Triangle * triangles;
				Geometry::sphere(triangles, triangle_count, attribute_stack.back().transform);

				MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(triangles, triangle_count);
				scene.add_mesh("Sphere", mesh_data_handle, attribute_stack.back().material);
			} else {
				WARNING(parser.location, "Unsupported Shape type '%.*s'!\n", unsigned(type.length()), type.start);
			}

		} else if (parser.match("WorldBegin")) {
			parser_skip(parser);

			if (attribute_stack.size() != 1) {
				ERROR(parser.location, "Invalid AttributeBegin block!\n");
			}
			attribute_stack.back().transform = Matrix4();

		} else if (parser.match("WorldEnd")) {
			parser_skip(parser);

		} else if (parser.match("Film") || parser.match("Sampler") || parser.match("Integrator")) { // Ingnore
			StringView type = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);

		} else {
			SourceLocation line_start = parser.location;
			const char   * line       = parser.cur;

			parser_next_line(parser);

			WARNING(line_start, "Skipped line: %.*s", unsigned(parser.cur - line), line);
		}
	}

	delete [] file;
}

void PBRTLoader::load(const char * filename, Scene & scene) {
	TextureMap  texture_map;
	MaterialMap material_map;
	MediumMap   medium_map;
	ObjectMap   object_map;

	char path[512]; Util::get_path(filename, path);
	int  path_length = strlen(path);

	load_include(filename, path, path_length, scene, texture_map, material_map, medium_map, object_map);
}