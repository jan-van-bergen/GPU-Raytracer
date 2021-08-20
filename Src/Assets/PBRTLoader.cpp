#include "PBRTLoader.h"

#include "Util/Util.h"
#include "Util/Parser.h"
#include "Util/Geometry.h"

struct Instance {
	MeshDataHandle mesh_data_handle;
	MaterialHandle material_handle;
};

using TextureMap  = HashMap<String, TextureHandle,   StringHash>;
using MaterialMap = HashMap<String, MaterialHandle,  StringHash>;
using ObjectMap   = HashMap<String, Array<Instance>, StringHash>;

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
		material.linear_roughness = sqrtf(find_param_float(params, "roughness", 0.5f));
	} else if (type == "dielectric") {
		material.type = Material::Type::DIELECTRIC;
	} else {
		printf("WARNING: Material Type '%.*s' is not supported!\n", unsigned(type.length()), type.start);
	}

	return material;
}

static void load_include(const char * filename, const char * path, int path_length, Scene & scene, TextureMap & texture_map, MaterialMap & material_map, ObjectMap & object_map) {
	int          file_length;
	const char * file = Util::file_read(filename, file_length);

	SourceLocation location;
	location.file = filename;
	location.line = 1;
	location.col  = 0;

	Parser parser;
	parser.init(file, file + file_length, location);

	struct Attribute {
		MaterialHandle current_material = MaterialHandle::get_default();

		struct {
			Vector3    position;
			Quaternion rotation;
			float      scale = 1.0f;
		} current_transform;
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
			load_include(include_abs, path, path_length, scene, texture_map, material_map, object_map);
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
				mesh.position = attribute_stack.back().current_transform.position;
				mesh.rotation = attribute_stack.back().current_transform.rotation;
				mesh.scale    = attribute_stack.back().current_transform.scale;
			}

		} else if (parser.match("Identity")) {
			parser_skip(parser);

			attribute_stack.back().current_transform.position = Vector3(0.0f);
			attribute_stack.back().current_transform.rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
			attribute_stack.back().current_transform.scale    = 1.0f;

		} else if (parser.match("Translate")) {
			parser_skip(parser);
			float x = parser.parse_float(); parser_skip(parser);
			float y = parser.parse_float(); parser_skip(parser);
			float z = parser.parse_float(); parser_skip(parser);

			attribute_stack.back().current_transform.position = Vector3(x, y, z);

		} else if (parser.match("Rotate")) {
			parser_skip(parser);
			float angle = Math::deg_to_rad(parser.parse_float()); parser_skip(parser);
			float x = parser.parse_float(); parser_skip(parser);
			float y = parser.parse_float(); parser_skip(parser);
			float z = parser.parse_float(); parser_skip(parser);

			attribute_stack.back().current_transform.rotation = Quaternion::axis_angle(Vector3(x, y, z), angle) * attribute_stack.back().current_transform.rotation;

		} else if (parser.match("Scale")) {
			parser_skip(parser);
			float x = parser.parse_float(); parser_skip(parser);
			float y = parser.parse_float(); parser_skip(parser);
			float z = parser.parse_float(); parser_skip(parser);

			attribute_stack.back().current_transform.scale = cbrtf(x * y * z); // Geometric mean

		} else if (parser.match("Transform")) {
			parser_skip(parser);
			parser.expect('[');
			parser_skip(parser);

			Matrix4 matrix;
			for (int i = 0; i < 16; i++) {
				matrix.cells[i] = parser.parse_float();
				parser_skip(parser);
			}

			parser.expect(']');
			parser_skip(parser);

			Matrix4::decompose(matrix,
				&attribute_stack.back().current_transform.position,
				&attribute_stack.back().current_transform.rotation,
				&attribute_stack.back().current_transform.scale,
				Vector3(0.0f, 0.0f, -1.0f)
			);

		} else if (parser.match("ConcatTransform")) {
			parser_skip(parser);
			parser.expect('[');
			parser_skip(parser);

			Matrix4 matrix;
			for (int i = 0; i < 16; i++) {
				matrix.cells[i] = parser.parse_float();
				parser_skip(parser);
			}

			parser.expect(']');
			parser_skip(parser);

			Vector3    position;
			Quaternion rotation;
			float      scale;
			Matrix4::decompose(matrix, &position, &rotation, &scale, Vector3(0.0f, 0.0f, -1.0f));

			attribute_stack.back().current_transform.position = Matrix4::transform_position(matrix, attribute_stack.back().current_transform.position);
			attribute_stack.back().current_transform.rotation = rotation * attribute_stack.back().current_transform.rotation;
			attribute_stack.back().current_transform.scale *= scale;

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

			attribute_stack.back().current_transform.position = Vector3(e_x, e_y, e_z);
			attribute_stack.back().current_transform.rotation = Quaternion::look_rotation(Vector3(f_x, f_y, f_z), Vector3(u_x, u_y, u_z));

		} else if (parser.match("Camera")) {
			StringView type = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);

			if (type == "perspective") {
				scene.camera.set_fov(Math::deg_to_rad(find_param_float(params, "fov", 90.0f)));
				scene.camera.aperture_radius = find_param_float(params, "lensradius", 0.1f);
				scene.camera.focal_distance  = find_param_float(params, "focaldistance", 10.0f);
			} else {
				WARNING(parser.location, "Unsupported Camera type '%.*s'!\m", unsigned(type.length()), type.start);
			}

			scene.camera.position = attribute_stack.back().current_transform.position;
			scene.camera.rotation = attribute_stack.back().current_transform.rotation;

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
			} else {
				WARNING(parser.location, "Unsupported texture class '%.*s'!\n", unsigned(clas.length()), clas.start);
			}

		} else if (parser.match("Material")) {
			StringView type = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);

			Material material = parse_material("Material", type, params, texture_map);
			attribute_stack.back().current_material = scene.asset_manager.add_material(material);

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

			if (!material_map.try_get(name, attribute_stack.back().current_material)) {
				attribute_stack.back().current_material = MaterialHandle::get_default();
				WARNING(parser.location, "Used undefined NamedMaterial '%.*s'!\n", unsigned(name.length()), name.start)
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

			attribute_stack.back().current_material = scene.asset_manager.add_material(material);

		} else if (parser.match("Shape")) {
			StringView type = parse_quoted(parser);
			parser_skip(parser);

			Array<Param> params = parse_params(parser);

			if (type == "plymesh") {
				StringView param_filename = find_param(params, "filename", Param::Type::STRING).strings[0];

				const char * filename_rel = param_filename.c_str();
				const char * filename_abs = get_absolute_filename(path, path_length, filename_rel, param_filename.length());

				MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(filename_abs);

				if (current_object.inside) {
					Instance instance = { };
					instance.mesh_data_handle = mesh_data_handle;
					instance.material_handle  = attribute_stack.back().current_material;
					current_object.shapes.push_back(instance);
				} else {
					Mesh & mesh = scene.add_mesh(filename_rel, mesh_data_handle, attribute_stack.back().current_material);
					mesh.position = attribute_stack.back().current_transform.position;
					mesh.rotation = attribute_stack.back().current_transform.rotation;
					mesh.scale    = attribute_stack.back().current_transform.scale;
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
					}

					triangle.init();
				}

				MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(triangles, triangle_count);
				scene.add_mesh("Triangle Mesh", mesh_data_handle, attribute_stack.back().current_material);
			} else if (type == "sphere") {
				float radius = find_param_float(params, "radius", 1.0f);

				Matrix4 transform =
					Matrix4::create_translation(attribute_stack.back().current_transform.position) *
					Matrix4::create_rotation   (attribute_stack.back().current_transform.rotation) *
					Matrix4::create_scale      (attribute_stack.back().current_transform.scale * radius);

				int        triangle_count;
				Triangle * triangles;
				Geometry::sphere(triangles, triangle_count, transform);

				MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(triangles, triangle_count);
				scene.add_mesh("Sphere", mesh_data_handle, attribute_stack.back().current_material);
			} else {
				WARNING(parser.location, "Unsupported Shape type '%.*s'!\n", unsigned(type.length()), type.start);
			}

		} else if (parser.match("WorldBegin")) {
			parser_skip(parser);

			if (attribute_stack.size() != 1) {
				ERROR(parser.location, "Invalid AttributeBegin block!\n");
			}
			attribute_stack.back().current_transform.position = Vector3(0.0f, 0.0f, 0.0f);
			attribute_stack.back().current_transform.rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
			attribute_stack.back().current_transform.scale    = 1.0f;

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
	ObjectMap   object_map;

	char path[512]; Util::get_path(filename, path);
	int  path_length = strlen(path);

	load_include(filename, path, path_length, scene, texture_map, material_map, object_map);
}
