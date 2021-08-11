#include "MitsubaLoader.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <unordered_map>
#include <charconv>

#include <miniz/miniz.h>

#include "MeshData.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

#include "Pathtracer/Scene.h"

#include "Util/Util.h"
#include "Util/Geometry.h"
#include "Util/StringView.h"

#define TAB_WIDTH 4

#define WARNING(loc, msg, ...) \
	printf("%s:%i:%i: " msg, loc.file, loc.line, loc.col, __VA_ARGS__);

#define ERROR(loc, msg, ...) \
	WARNING(loc, msg, __VA_ARGS__); \
	abort();

static bool is_whitespace(char c) {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

struct SourceLocation {
	const char * file;
	int          line;
	int          col;

	void advance(char c) {
		if (c == '\n') {
			line++;
			col = 0;
		} else if (c == '\t') {
			col += TAB_WIDTH;
		} else {
			col++;
		}
	}
};

struct ParserState {
	const char * cur;
	const char * end;

	SourceLocation location;

	void init(const char * cur, const char * end, SourceLocation location) {
		this->cur = cur;
		this->end = end;
		this->location = location;
	}

	bool reached_end() const {
		return cur >= end;
	}

	void advance() {
		location.advance(*cur);
		cur++;
	}

	void skip_whitespace() {
		while (cur < end && is_whitespace(*cur)) advance();
	}

	bool match(char target) {
		if (cur < end && *cur == target) {
			advance();
			return true;
		}

		return false;
	}

	void expect(char expected) {
		if (reached_end()) {
			ERROR(location, "End of File!\n");
		}
		if (*cur != expected) {
			ERROR(location, "Unexpected char '%c', expected '%c'!\n", *cur, expected)
		}
		advance();
	}

	template<typename T>
	T parse_number() {
		T number = { }; 
		std::from_chars_result result = std::from_chars(cur, end, number);

		if (bool(result.ec)) {
			ERROR(location, "Failed to parse '%.*s' as a number!\n", unsigned(end - cur), cur);
		}

		cur = result.ptr;
		return number;
	}
};

struct XMLAttribute {
	StringView name;
	StringView value;

	SourceLocation location_of_value;

	template<typename T> T get_value() const;

	template<>
	int get_value() const {
		ParserState parser = { };
		parser.init(value.start, value.end, location_of_value);
		return parser.parse_number<int>();
	}
	
	template<>
	float get_value() const {
		ParserState parser = { };
		parser.init(value.start, value.end, location_of_value);
		return parser.parse_number<float>();
	}
	
	template<>
	bool get_value() const {
		if (value == "true")  return true;
		if (value == "false") return false;
		ERROR(location_of_value, "Unable to parse '%.*s' as boolean!\n", unsigned(value.length()), value.start);
	}

	template<>
	Vector3 get_value() const {
		ParserState parser = { };
		parser.init(value.start, value.end, location_of_value);

		Vector3 v;
		v.x = parser.parse_number<float>();

		if (parser.match(',')) {
			parser.match(' ');

			v.y = parser.parse_number<float>();

			parser.expect(',');
			parser.match(' ');

			v.z = parser.parse_number<float>();
		} else {
			v.y = v.x;
			v.z = v.x;
		}

		return v;
	}

	template<>
	Matrix4 get_value() const {
		ParserState parser = { };
		parser.init(value.start, value.end, location_of_value);

		int i = 0;
		Matrix4 m;
		while (true) {
			m.cells[i] = parser.parse_number<float>();

			if (++i == 16) break;
			
			parser.expect(' ');
		}

		return m;
	}
};

struct XMLNode {
	StringView tag;

	bool is_question_mark;

	std::vector<XMLAttribute> attributes;
	std::vector<XMLNode>      children;

	SourceLocation location;

	template<typename Predicate>
	const XMLAttribute * find_attribute(Predicate predicate) const {
		for (int i = 0; i < attributes.size(); i++) {
			if (predicate(attributes[i])) {
				return &attributes[i];
			}
		}
		return nullptr;
	}
	
	template<typename Predicate>
	const XMLNode * find_child(Predicate predicate) const {
		for (int i = 0; i < children.size(); i++) {
			if (predicate(children[i])) {
				return &children[i];
			}
		}
		return nullptr;
	}
	
	const XMLAttribute * find_attribute(const char * name) const {
		return find_attribute([name](const XMLAttribute & attribute) { return attribute.name == name; });
	}
	
	template<typename T>
	T get_optional_attribute(const char * name, T default_value) const {
		const XMLAttribute * attribute = find_attribute([name](const XMLAttribute & attribute) { return attribute.name == name; });
		if (attribute) {
			return attribute->get_value<T>();
		} else {
			return default_value;
		}
	}
	
	const XMLNode * find_child(const char * tag) const {
		return find_child([tag](const XMLNode & node) { return node.tag == tag; });
	}
	
	const XMLNode * find_child_by_name(const char * name) const {
		return find_child([name](const XMLNode & node) {
			const XMLAttribute * attr = node.find_attribute("name");
			if (attr == nullptr) return false;

			return attr->value == name;
		});
	}
	template<typename T>
	T get_optional_child_value(const char * name, T default_value) const {
		const XMLNode * child = find_child_by_name(name);
		if (child) {
			return child->get_optional_attribute("value", default_value);
		} else {
			return default_value;
		}
	}
};

static XMLNode parse_tag(ParserState & parser) {
	if (parser.reached_end()) return { };

	parser.expect('<');

	// Parse Comment
	while (parser.match('!')) {
		parser.expect('-');
		parser.expect('-');

		while (!parser.reached_end()) {
			if (parser.match('-') && parser.match('-') && parser.match('>')) break;
			parser.advance();
		}

		parser.skip_whitespace();
		parser.expect('<');
	}
		
	XMLNode node = { };
	node.location         = parser.location;
	node.is_question_mark = parser.match('?');

	// Parse node tag
	node.tag.start = parser.cur;
	while (!parser.reached_end() && !is_whitespace(*parser.cur)) parser.advance();
	node.tag.end = parser.cur;

	parser.skip_whitespace();

	// Parse attributes
	while (!parser.reached_end() && !parser.match('>')) {
		XMLAttribute attribute = { };
		
		// Parse attribute name
		attribute.name.start = parser.cur;
		while (!parser.reached_end() && *parser.cur != '=') parser.advance();
		attribute.name.end = parser.cur;

		parser.expect('=');

		char quote_type; // Either single or double quotes
		if (parser.match('"')) {
			quote_type = '"';
		} else if (parser.match('\'')) {
			quote_type = '\'';
		} else {
			ERROR(parser.location, "An attribute must begin with either double or single quotes!\n");
		}

		attribute.location_of_value = parser.location;

		// Parse attribute value
		attribute.value.start = parser.cur;
		while (!parser.reached_end() && *parser.cur != quote_type) parser.advance();
		attribute.value.end = parser.cur;

		parser.expect(quote_type);
		parser.skip_whitespace();

		node.attributes.push_back(attribute);

		// Check if this is an inline tag (i.e. <tag/> or <?tag?>), if so return
		if (parser.match('/') || (node.is_question_mark && parser.match('?'))) {
			parser.expect('>');
			return node;
		}
	}

	parser.skip_whitespace();

	// Parse children
	do {
		node.children.push_back(parse_tag(parser));
		parser.skip_whitespace();
	} while (!(parser.cur + 1 < parser.end && parser.cur[0] == '<' && parser.cur[1] == '/'));

	parser.expect('<');
	parser.expect('/');

	int i = 0;
	while (!parser.reached_end() && !parser.match('>')) {
		if (*parser.cur != node.tag.start[i++]) {
			ERROR(parser.location, "Non matching closing tag for '%.*s'!\n", unsigned(node.tag.length()), node.tag.start);
		}
		parser.advance();
	}

	return node;
}

static XMLNode parse_xml(ParserState & parser) {
	XMLNode node;

	do {
		parser.skip_whitespace();
		node = parse_tag(parser);
	} while (node.is_question_mark);

	return node;
}

struct ShapeGroup {
	MeshDataHandle mesh_data_handle;
	MaterialHandle material_handle;
};

struct Serialized {
	uint32_t         num_meshes;
	const char    ** mesh_names;
	MeshDataHandle * mesh_data_handles;
};

using ShapeGroupMap = HashMap<StringView, ShapeGroup,     StringView::Hash>;
using SerializedMap = HashMap<StringView, Serialized,     StringView::Hash>;
using MaterialMap   = HashMap<StringView, MaterialHandle, StringView::Hash>;
using TextureMap    = HashMap<StringView, TextureHandle,  StringView::Hash>;

static const char * get_absolute_filename(const char * path, int len_path, const char * filename, int len_filename) {
	char * filename_abs = new char[len_path + len_filename + 1];

	memcpy(filename_abs,            path,     len_path);
	memcpy(filename_abs + len_path, filename, len_filename);
	filename_abs[len_path + len_filename] = '\0';

	return filename_abs;
}

static void parse_rgb_or_texture(const XMLNode * node, const char * name, const TextureMap & texture_map, const char * path, Scene & scene, Vector3 & rgb, TextureHandle & texture) {
	const XMLNode * reflectance = node->find_child_by_name(name);
	if (reflectance) {
		if (reflectance->tag == "rgb") {
			rgb = reflectance->get_optional_attribute("value", Vector3(1.0f));
			return;
		} else if (reflectance->tag == "srgb") {
			rgb = reflectance->get_optional_attribute("value", Vector3(1.0f));
			rgb.x = Math::gamma_to_linear(rgb.x);
			rgb.y = Math::gamma_to_linear(rgb.y);
			rgb.z = Math::gamma_to_linear(rgb.z);
			return;
		} else if (reflectance->tag == "texture") {
			const StringView & filename_rel = reflectance->find_child_by_name("filename")->find_attribute("value")->value;
			const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

			texture = scene.asset_manager.add_texture(filename_abs);

			const XMLNode * scale = reflectance->find_child_by_name("scale");
			if (scale) {
				rgb = scale->get_optional_attribute("value", Vector3(1.0f));
			}

			delete [] filename_abs;
			return;
		} else if (reflectance->tag == "ref") {
			const StringView & texture_name = reflectance->find_attribute("id")->value;
			bool found = texture_map.try_get(texture_name, texture);
			if (!found) {
				WARNING(reflectance->location, "Invalid texture ref '%.*s'!", unsigned(texture_name.length()), texture_name.start);
			}
		}
	}
	rgb = Vector3(1.0f);
}

static void parse_transform(const XMLNode * node, Vector3 * position, Quaternion * rotation, float * scale, const Vector3 & forward = Vector3(0.0f, 0.0f, 1.0f)) {
	const XMLNode * transform = node->find_child("transform");
	if (transform) {
		const XMLNode * matrix = transform->find_child("matrix");
		if (matrix) {
			Matrix4 world = matrix->find_attribute("value")->get_value<Matrix4>();

			// Decompose Matrix
			if (position) *position = Vector3(world(0, 3), world(1, 3), world(2, 3));
	
			if (rotation) *rotation = Quaternion::look_rotation(
				Matrix4::transform_direction(world, forward),
				Matrix4::transform_direction(world, Vector3(0.0f, 1.0f, 0.0f))
			);

			if (scale) {
				float scale_x = Vector3::length(Vector3(world(0, 0), world(0, 1), world(0, 2)));
				float scale_y = Vector3::length(Vector3(world(1, 0), world(1, 1), world(1, 2)));
				float scale_z = Vector3::length(Vector3(world(2, 0), world(2, 1), world(2, 2)));

				if (Math::approx_equal(scale_x, scale_y) && Math::approx_equal(scale_y, scale_z)) {
					*scale = scale_x;
				} else {
					WARNING(matrix->location, "Warning: non-uniform scaling is not supported!\n");
					*scale = cbrt(scale_x * scale_y * scale_z);
				}
			}

			return;
		}
		
		const XMLNode * lookat = transform->find_child("lookat");
		if (lookat) {
			Vector3 origin = lookat->get_optional_attribute("origin", Vector3(0.0f, 0.0f,  0.0f));
			Vector3 target = lookat->get_optional_attribute("target", Vector3(0.0f, 0.0f, -1.0f));
			Vector3 up     = lookat->get_optional_attribute("up",     Vector3(0.0f, 1.0f,  0.0f));

			Vector3 forward = Vector3::normalize(target - origin);

			if (position) *position = origin;
			if (rotation) *rotation = Quaternion::look_rotation(forward, up);
		}

		const XMLNode * scale_node = transform->find_child("scale");
		if (scale_node && scale) {
			const XMLAttribute * scale_value = scale_node->find_attribute("value");
			if (scale_value) {
				*scale = scale_value->get_value<float>();
			} else {
				float x = scale_node->get_optional_attribute("x", 1.0f);
				float y = scale_node->get_optional_attribute("y", 1.0f);
				float z = scale_node->get_optional_attribute("z", 1.0f);

				*scale = x * y * z;
			}
		}
		
		const XMLNode * rotate = transform->find_child("rotate");
		if (rotate && rotation) {
			float x = rotate->get_optional_attribute("x", 0.0f);
			float y = rotate->get_optional_attribute("y", 0.0f);
			float z = rotate->get_optional_attribute("z", 0.0f);

			if (x == 0.0f && y == 0.0f && z == 0.0f) {
				WARNING(rotate->location, "WARNING: Rotation without axis specified!\n");
			} else {
				float angle = rotate->get_optional_attribute("angle", 0.0f);
				*rotation = Quaternion::axis_angle(Vector3(x, y, z), Math::deg_to_rad(angle));
			}
		}

		const XMLNode * translate = transform->find_child("translate");
		if (translate && position) {
			*position = Vector3(
				translate->get_optional_attribute("x", 0.0f),
				translate->get_optional_attribute("y", 0.0f),
				translate->get_optional_attribute("z", 0.0f)
			);
		}
	} else {
		if (position) *position = Vector3(0.0f);
		if (rotation) *rotation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
		if (scale)    *scale    = 1.0f;
	}
}

static Matrix4 parse_transform_matrix(const XMLNode * node) {
	const XMLNode * transform = node->find_child("transform");
	if (transform) {
		const XMLNode * matrix = transform->find_child("matrix");
		if (matrix) {
			return matrix->find_attribute("value")->get_value<Matrix4>();
		}
	}

	Vector3    translation;
	Quaternion rotation;
	float      scale = 1.0f;
	parse_transform(node, &translation, &rotation, &scale);

	return Matrix4::create_translation(translation) * Matrix4::create_rotation(rotation) * Matrix4::create_scale(scale);
}

static MaterialHandle parse_material(const XMLNode * node, Scene & scene, const MaterialMap & material_map, const TextureMap & texture_map, const char * path) {
	Material material = { };

	const XMLNode * bsdf;

	if (node->tag != "bsdf") {
		// Check if there is an emitter defined
		const XMLNode * emitter = node->find_child("emitter");
		if (emitter) {
			material.type = Material::Type::LIGHT;
			material.name = "emitter";
			material.emission = emitter->find_child_by_name("radiance")->find_attribute("value")->get_value<Vector3>();

			return scene.asset_manager.add_material(material);
		}

		// Check if an existing Material is referenced
		const XMLNode * ref = node->find_child("ref");
		if (ref) {
			const StringView & material_name = ref->find_attribute("id")->value;
		
			MaterialHandle material_id;
			bool found = material_map.try_get(material_name, material_id);
			if (!found) {
				WARNING(ref->location, "Invalid material Ref '%.*s'!\n", unsigned(material_name.length()), material_name.start);
			
				return MaterialHandle::get_default();
			}

			return material_id;
		}
		
		// Otherwise, parse BSDF
		bsdf = node->find_child("bsdf");
		if (bsdf == nullptr) {
			WARNING(node->location, "Unable to parse BSDF!\n");
			return MaterialHandle::get_default();
		}
	} else {
		bsdf = node;
	}

	const XMLAttribute * name = bsdf->find_attribute("id");
		
	const XMLNode      * inner_bsdf = bsdf;
	const XMLAttribute * inner_bsdf_type = inner_bsdf->find_attribute("type");

	// Keep peeling back nested BSDFs, we only care about the innermost one
	while (
		inner_bsdf_type->value == "twosided" ||
		inner_bsdf_type->value == "mask" ||
		inner_bsdf_type->value == "bumpmap" ||
		inner_bsdf_type->value == "coating"
	) {
		inner_bsdf      = inner_bsdf->find_child("bsdf");
		inner_bsdf_type = inner_bsdf->find_attribute("type");

		if (name == nullptr) {
			name = inner_bsdf->find_attribute("id");
		} 
	}

	if (name) {
		material.name = name->value.c_str();
	} else {
		material.name = "Material";
	}
		
	if (inner_bsdf_type->value == "diffuse") {
		material.type = Material::Type::DIFFUSE;
				
		parse_rgb_or_texture(inner_bsdf, "reflectance", texture_map, path, scene, material.diffuse, material.texture_id);
	} else if (inner_bsdf_type->value == "conductor") {
		material.type = Material::Type::GLOSSY;

		parse_rgb_or_texture(inner_bsdf, "specularReflectance", texture_map, path, scene, material.diffuse, material.texture_id);

		material.linear_roughness    = 0.0f;
		material.index_of_refraction = bsdf->get_optional_child_value("eta", 1.0f);
	} else if (inner_bsdf_type->value == "roughconductor" || inner_bsdf_type->value == "roughdiffuse") {
		material.type = Material::Type::GLOSSY;

		parse_rgb_or_texture(inner_bsdf, "specularReflectance", texture_map, path, scene, material.diffuse, material.texture_id);

		material.linear_roughness    = bsdf->get_optional_child_value("alpha", 0.5f);
		material.index_of_refraction = bsdf->get_optional_child_value("eta",   1.0f);
	} else if (inner_bsdf_type->value == "plastic" || inner_bsdf_type->value == "roughplastic") {
		material.type = Material::Type::GLOSSY;

		parse_rgb_or_texture(inner_bsdf, "diffuseReflectance", texture_map, path, scene, material.diffuse, material.texture_id);

		material.linear_roughness    = bsdf->get_optional_child_value("alpha",  0.5f);
		material.index_of_refraction = bsdf->get_optional_child_value("intIOR", 1.0f);

		const XMLNode * nonlinear = inner_bsdf->find_child_by_name("nonlinear");
		if (nonlinear && nonlinear->find_attribute("value")->get_value<bool>()) {
			material.linear_roughness = sqrtf(material.linear_roughness);
		}
	} else if (inner_bsdf_type->value == "thindielectric" || inner_bsdf_type->value == "dielectric" || inner_bsdf_type->value == "roughdielectric") {
		material.type = Material::Type::DIELECTRIC;
		material.transmittance = Vector3(1.0f);
		material.index_of_refraction = bsdf->get_optional_child_value("intIOR", 1.333f);
	} else if (inner_bsdf_type->value == "difftrans") {
		material.type = Material::Type::DIFFUSE;
				
		parse_rgb_or_texture(inner_bsdf, "transmittance", texture_map, path, scene, material.diffuse, material.texture_id);
	} else {
		WARNING(inner_bsdf_type->location_of_value, "WARNING: BSDF type '%.*s' not supported!\n", unsigned(inner_bsdf_type->value.length()), inner_bsdf_type->value.start);
			
		return MaterialHandle::get_default();
	}

	return scene.asset_manager.add_material(material);
}

static Serialized parse_serialized(const XMLNode * node, const char * filename, Scene & scene) {
	int          serialized_length;
	const char * serialized = Util::file_read(filename, serialized_length);
		
	uint16_t file_format_id; memcpy(&file_format_id, serialized,                    sizeof(uint16_t));
	uint16_t file_version;   memcpy(&file_version,   serialized + sizeof(uint16_t), sizeof(uint16_t));
	
	if (file_format_id != 0x041c) {
		ERROR(node->location, "ERROR: Serialized file '%s' does not start with format ID 0x041c!\n", filename);
	}

	// Read the End-of-File Dictionary
	uint32_t num_meshes; memcpy(&num_meshes, serialized + serialized_length - sizeof(uint32_t), sizeof(uint32_t));

	uint64_t eof_dictionary_offset = serialized_length - sizeof(uint32_t) - (num_meshes - 1) * sizeof(uint64_t) - 8;

	uint64_t * mesh_offsets = new uint64_t[num_meshes + 1];
	memcpy(mesh_offsets, serialized + eof_dictionary_offset, num_meshes * sizeof(uint64_t));
	mesh_offsets[num_meshes] = serialized_length - sizeof(uint32_t);
	assert(mesh_offsets[0] == 0);

	const char    ** mesh_names        = new const char * [num_meshes];
	MeshDataHandle * mesh_data_handles = new MeshDataHandle[num_meshes];

	for (uint32_t i = 0; i < num_meshes; i++) {
		// Decompress stream for this Mesh
		mz_ulong num_bytes = mesh_offsets[i+1] - mesh_offsets[i] - 4;

		uLong  deserialized_length;
		char * deserialized;

		while (true) {
			deserialized_length = compressBound(num_bytes);
			deserialized = new char[deserialized_length];

			int status = uncompress(
				reinterpret_cast<      unsigned char *>(deserialized), &deserialized_length,
				reinterpret_cast<const unsigned char *>(serialized + mesh_offsets[i] + 4), num_bytes
			);

			if (status == MZ_BUF_ERROR) {
				delete [] deserialized;
				num_bytes *= 2;
			} else if (status == MZ_OK) {
				break;
			} else {
				ERROR(node->location, "ERROR: Failed to decompress file '%s'!\n", filename);
			}
		} 

		char const * cur = deserialized;

		// Helper function to copy over n bytes from the stream
		auto read_n = [&cur, deserialized, deserialized_length, node](void * dst, size_t num_bytes) {
			size_t offset = cur - deserialized;
			if (offset + num_bytes > deserialized_length) {
				ERROR(node->location, "ERROR: Buffer overflow!");
			}

			memcpy(dst, cur, num_bytes);
			cur += num_bytes;
		};

		// Read flags field
		uint32_t flags; read_n(&flags, sizeof(uint32_t));
		bool flag_has_normals      = flags & 0x0001;
		bool flag_has_tex_coords   = flags & 0x0002;
		bool flag_has_colours      = flags & 0x0008;
		bool flag_use_face_normals = flags & 0x0010;
		bool flag_single_precision = flags & 0x1000;
		bool flag_double_precision = flags & 0x2000;

		// Read null terminated name
		const char * name_start = cur;
		cur += strlen(cur) + 1;

		size_t name_length = cur - name_start;
		char * mesh_name = new char[name_length];
		memcpy(mesh_name, name_start, name_length);
		mesh_names[i] = mesh_name;

		// Read number of vertices and triangles
		uint64_t num_vertices;  read_n(&num_vertices,  sizeof(uint64_t));
		uint64_t num_triangles;	read_n(&num_triangles, sizeof(uint64_t));
		
		if (num_vertices == 0 || num_triangles == 0) {
			mesh_data_handles[i] = MeshDataHandle { INVALID };
			
			WARNING(node->location, "WARNING: Serialized Mesh defined without vertices or triangles, skipping\n");

			delete [] deserialized;
			continue;
		}
		
		// Check if num_vertices fits inside a uint32_t to determine whether indices use 32 or 64 bits
		bool fits_in_32_bits = num_vertices <= 0xffffffff;

		size_t element_size;
		if (flag_single_precision) {
			element_size = sizeof(float);
		} else if (flag_double_precision) {
			element_size = sizeof(double);
		} else {
			ERROR(node->location, "ERROR: Neither single nor double precision specified!\n");
		}

		const char * vertex_positions = cur;
		cur += num_vertices * 3 * element_size;

		const char * vertex_normals = cur;
		if (flag_has_normals) cur += num_vertices * 3 * element_size;
		
		const char * vertex_tex_coords = cur;
		if (flag_has_tex_coords) cur += num_vertices * 2 * element_size;
		
		// Vertex colours, not used
		if (flag_has_colours) cur += num_vertices * 3 * element_size;
		
		const char * indices = cur;

		// Reads a Vector3 from a buffer with the appropriate precision
		auto read_vector3 = [flag_single_precision](const char * buffer, uint64_t index) {
			Vector3 result;
			if (flag_single_precision) {
				result.x = reinterpret_cast<const float *>(buffer)[3*index];
				result.y = reinterpret_cast<const float *>(buffer)[3*index + 1];
				result.z = reinterpret_cast<const float *>(buffer)[3*index + 2];
			} else {
				result.x = reinterpret_cast<const double *>(buffer)[3*index];
				result.y = reinterpret_cast<const double *>(buffer)[3*index + 1];
				result.z = reinterpret_cast<const double *>(buffer)[3*index + 2];
			}
			return result;
		};
		
		// Reads a Vector2 from a buffer with the appropriate precision
		auto read_vector2 = [flag_single_precision](const char * buffer, uint64_t index) {
			Vector2 result;
			if (flag_single_precision) {
				result.x = reinterpret_cast<const float *>(buffer)[2*index];
				result.y = reinterpret_cast<const float *>(buffer)[2*index + 1];
			} else {
				result.x = reinterpret_cast<const double *>(buffer)[2*index];
				result.y = reinterpret_cast<const double *>(buffer)[2*index + 1];
			}
			return result;
		};
		
		// Reads a 32 or 64 bit indx from the specified buffer
		auto read_index = [fits_in_32_bits](const char * buffer, uint64_t index) -> uint64_t {
			if (fits_in_32_bits) {
				return reinterpret_cast<const uint32_t *>(buffer)[index];
			} else {
				return reinterpret_cast<const uint64_t *>(buffer)[index];
			}
		};
		
		// Construct triangles
		Triangle * triangles = new Triangle[num_triangles];
		
		for (size_t t = 0; t < num_triangles; t++) {
			uint64_t index_0 = read_index(indices, 3*t);
			uint64_t index_1 = read_index(indices, 3*t + 1);
			uint64_t index_2 = read_index(indices, 3*t + 2);

			triangles[t].position_0 = read_vector3(vertex_positions, index_0);
			triangles[t].position_1 = read_vector3(vertex_positions, index_1);
			triangles[t].position_2 = read_vector3(vertex_positions, index_2);

			if (flag_use_face_normals) {
				Vector3 geometric_normal = Vector3::normalize(Vector3::cross(
					triangles[t].position_1 - triangles[t].position_0,
					triangles[t].position_2 - triangles[t].position_0
				));
				triangles[t].normal_0 = geometric_normal;
				triangles[t].normal_1 = geometric_normal;
				triangles[t].normal_2 = geometric_normal;
			} else if (flag_has_normals) {
				triangles[t].normal_0 = read_vector3(vertex_normals, index_0);
				triangles[t].normal_1 = read_vector3(vertex_normals, index_1);
				triangles[t].normal_2 = read_vector3(vertex_normals, index_2);
			}

			if (flag_has_tex_coords) {
				triangles[t].tex_coord_0 = read_vector2(vertex_tex_coords, index_0);
				triangles[t].tex_coord_1 = read_vector2(vertex_tex_coords, index_1);
				triangles[t].tex_coord_2 = read_vector2(vertex_tex_coords, index_2);
			}

			triangles[t].calc_aabb();
		}

		mesh_data_handles[i] = scene.asset_manager.add_mesh_data(triangles, num_triangles);

		delete [] deserialized;
	}

	delete [] mesh_offsets;
	delete [] serialized;
	
	Serialized result = { };
	result.num_meshes        = num_meshes;
	result.mesh_names        = mesh_names;
	result.mesh_data_handles = mesh_data_handles;

	return result;
}

static MeshDataHandle parse_shape(const XMLNode * node, Scene & scene, SerializedMap & serialized_map, const char * path, const char *& name) {
	const XMLAttribute * type = node->find_attribute("type");
	if (type->value == "obj") {
		const StringView & filename_rel = node->find_child_by_name("filename")->find_attribute("value")->value;
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(filename_abs);
		delete [] filename_abs;

		name = filename_rel.c_str();
		
		return mesh_data_handle;
	} else if (type->value == "rectangle" || type->value == "cube" || type->value == "disk" || type->value == "sphere") {
		Matrix4 world = parse_transform_matrix(node);

		Triangle * triangles = nullptr;
		int        triangle_count = 0;

		if (type->value == "rectangle") {
			Geometry::rectangle(triangles, triangle_count, world);
		} else if (type->value == "cube") {
			Geometry::cube(triangles, triangle_count, world);
		} else if (type->value == "disk") {
			Geometry::disk(triangles, triangle_count, world);
		} else if (type->value == "sphere") {
			float   radius = node->get_optional_child_value("radius", 1.0f);
			Vector3 center = Vector3(0.0f);

			const XMLNode * xml_center = node->find_child_by_name("center");
			if (xml_center) {
				center = Vector3(
					xml_center->get_optional_attribute("x", 0.0f),
					xml_center->get_optional_attribute("y", 0.0f),
					xml_center->get_optional_attribute("z", 0.0f)
				);
			}
				
			world = world * Matrix4::create_translation(center) * Matrix4::create_scale(radius);

			Geometry::sphere(triangles, triangle_count, world);
		}

		name = type->value.c_str();

		return scene.asset_manager.add_mesh_data(triangles, triangle_count);
	} else if (type->value == "serialized") {
		const StringView & filename_rel = node->find_child_by_name("filename")->find_attribute("value")->value;

		Serialized serialized;
		bool found = serialized_map.try_get(filename_rel, serialized);
		if (!found) {
			const char * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());
		
			serialized = parse_serialized(node, filename_abs, scene);
			serialized_map[filename_rel] = serialized;

			delete [] filename_abs;
		}

		int shape_index = node->get_optional_child_value("shapeIndex", 0);

		name = serialized.mesh_names[shape_index];
		return serialized.mesh_data_handles[shape_index];
	} else {
		WARNING(node->location, "WARNING: Shape type '%.*s' not supported!\n", unsigned(type->value.length()), type->value.start);
		return MeshDataHandle { INVALID };
	}
}

static void walk_xml_tree(const XMLNode * node, Scene & scene, ShapeGroupMap & shape_group_map, SerializedMap & serialized_map, MaterialMap & material_map, TextureMap & texture_map, const char * path) {
	if (node->tag == "bsdf") {
		MaterialHandle   material_handle = parse_material(node, scene, material_map, texture_map, path);
		const Material & material = scene.asset_manager.get_material(material_handle);

		StringView str = { material.name, material.name + strlen(material.name) };
		material_map[str] = material_handle; 
	} else if (node->tag == "texture") {
		const StringView & filename_rel = node->find_child_by_name("filename")->find_attribute("value")->value;
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		StringView texture_id = { };
		const XMLAttribute * id = node->find_attribute("id");
		if (id) {
			texture_id = id->value;
		} else {
			texture_id = filename_rel;
		}
		texture_map[texture_id] = scene.asset_manager.add_texture(filename_abs);

		delete [] filename_abs;
	} else if (node->tag == "shape") {
		const XMLAttribute * type = node->find_attribute("type");
		if (type->value == "obj" || type->value == "rectangle" || type->value == "cube" || type->value == "disk" || type->value == "sphere" || type->value == "serialized") {
			char const * name = nullptr;
			MeshDataHandle mesh_data_handle = parse_shape(node, scene, serialized_map, path, name);
			MaterialHandle material_handle  = parse_material(node, scene, material_map, texture_map, path);

			if (mesh_data_handle.handle != INVALID) {
				Mesh & mesh = scene.add_mesh(name, mesh_data_handle, material_handle);

				// Do not apply transform to primitive shapes, since they have the transform baked into their vertices
				if (type->value == "obj" || type->value == "serialized") { 
					parse_transform(node, &mesh.position, &mesh.rotation, &mesh.scale);
				}
			}
		} else if (type->value == "shapegroup") {
			const XMLNode * shape = node->find_child("shape");

			const char * name = nullptr;
			MeshDataHandle mesh_data_handle = parse_shape(shape, scene, serialized_map, path, name);
			MaterialHandle material_handle  = parse_material(shape, scene, material_map, texture_map, path);
			
			const StringView & id = node->find_attribute("id")->value;
			shape_group_map[id] = { mesh_data_handle, material_handle };
		} else if (type->value == "instance") {
			const XMLNode      * ref = node->find_child("ref");
			const XMLAttribute * id  = ref->find_attribute("id");

			ShapeGroup shape_group;
			if (shape_group_map.try_get(id->value, shape_group) && shape_group.mesh_data_handle.handle != INVALID) {
				Mesh & mesh = scene.add_mesh(id->value.c_str(), shape_group.mesh_data_handle, shape_group.material_handle);
				parse_transform(node, &mesh.position, &mesh.rotation, &mesh.scale);
			}
		} else {
			WARNING(node->location, "WARNING: Shape type '%.*s' not supported!\n", unsigned(type->value.length()), type->value.start);
		}
	} else if (node->tag == "sensor") {
		const StringView & camera_type = node->find_attribute("type")->value;

		if (camera_type == "perspective" || camera_type == "perspective_rdist" || camera_type == "thinlens") {
			float fov = node->get_optional_child_value("fov", 110.0f);
			scene.camera.set_fov(Math::deg_to_rad(fov));
			scene.camera.aperture_radius = node->get_optional_child_value("aperatureRadius", 0.05f);
			scene.camera.focal_distance  = node->get_optional_child_value("focusDistance", 10.0f);

			float scale = 1.0f;
			parse_transform(node, &scene.camera.position, &scene.camera.rotation, &scale, Vector3(0.0f, 0.0f, -1.0f));
			
			if (scale < 0.0f) {
				scene.camera.rotation = Quaternion::conjugate(scene.camera.rotation);
			}
		} else {
			WARNING(node->location, "WARNING: Camera type '%.*s' not supported!\n", unsigned(camera_type.length()), camera_type.start);
		}
	} else if (node->tag == "include") {
		const StringView & filename_rel = node->find_attribute("filename")->value;
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		MitsubaLoader::load(filename_abs, scene);

		delete [] filename_abs;
	} else for (int i = 0; i < node->children.size(); i++) {
		walk_xml_tree(&node->children[i], scene, shape_group_map, serialized_map, material_map, texture_map, path);
	}
}

void MitsubaLoader::load(const char * filename, Scene & scene) {
	int          source_length;
	const char * source = Util::file_read(filename, source_length);

	SourceLocation location = { };
	location.file = filename;
	location.line = 1;
	location.col  = 0;

	ParserState parser = { };
	parser.init(source, source + source_length, location);

	XMLNode root = parse_xml(parser);

	ShapeGroupMap shape_group_map;
	SerializedMap serialized_map;
	MaterialMap   material_map;
	TextureMap    texture_map;
	char path[512];	Util::get_path(filename, path);
	walk_xml_tree(&root, scene, shape_group_map, serialized_map, material_map, texture_map, path);

	delete [] source;
}
