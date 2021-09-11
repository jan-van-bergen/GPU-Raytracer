#include "MitsubaLoader.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <miniz/miniz.h>

#include "MeshData.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

#include "Pathtracer/Scene.h"

#include "Util/Util.h"
#include "Util/Array.h"
#include "Util/HashMap.h"
#include "Util/Parser.h"
#include "Util/Geometry.h"
#include "Util/StringView.h"

struct XMLAttribute {
	StringView name;
	StringView value;

	SourceLocation location_of_value;

	template<typename T> T get_value() const;

	template<>
	StringView get_value() const {
		return value;
	}

	template<>
	int get_value() const {
		Parser parser = { };
		parser.init(value.start, value.end, location_of_value);
		return parser.parse_int();
	}

	template<>
	float get_value() const {
		Parser parser = { };
		parser.init(value.start, value.end, location_of_value);
		return parser.parse_float();
	}

	template<>
	bool get_value() const {
		if (value == "true")  return true;
		if (value == "false") return false;
		ERROR(location_of_value, "Unable to parse '%.*s' as boolean!\n", unsigned(value.length()), value.start);
	}

	template<>
	Vector3 get_value() const {
		Parser parser = { };
		parser.init(value.start, value.end, location_of_value);

		Vector3 v;
		v.x = parser.parse_float();

		if (parser.match(',')) {
			parser.match(' ');

			v.y = parser.parse_float();

			parser.expect(',');
			parser.match(' ');

			v.z = parser.parse_float();
		} else {
			v.y = v.x;
			v.z = v.x;
		}

		return v;
	}

	template<>
	Matrix4 get_value() const {
		Parser parser = { };
		parser.init(value.start, value.end, location_of_value);

		int i = 0;
		Matrix4 m;
		while (true) {
			m.cells[i] = parser.parse_float();

			if (++i == 16) break;

			parser.expect(' ');
		}

		return m;
	}
};

struct XMLNode {
	StringView tag;

	bool is_question_mark;

	Array<XMLAttribute> attributes;
	Array<XMLNode>      children;

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
	T get_attribute_value(const char * name) const {
		const XMLAttribute * attribute = find_attribute(name);
		if (attribute) {
			return attribute->get_value<T>();
		} else {
			ERROR(location, "Node '%.*s' does not have an attribute with name '%s'!\n", unsigned(tag.length()), tag.start, name);
		}
	}

	template<typename T>
	T get_child_value(const char * child_name, const char * attribute_name = "value") const {
		const XMLNode * child = find_child_by_name(child_name);
		if (child) {
			return child->get_attribute_value<T>(attribute_name);
		} else {
			ERROR(location, "Node '%.*s' does not have a child with name '%s'!\n", unsigned(tag.length()), tag.start, child_name);
		}
	}

	template<typename T>
	T get_child_value_optional(const char * name, T default_value) const {
		const XMLNode * child = find_child_by_name(name);
		if (child) {
			return child->get_optional_attribute("value", default_value);
		} else {
			return default_value;
		}
	}
};

static void parser_skip(Parser & parser) {
	parser.skip_whitespace_or_newline();

	while (parser.match("<!--")) {
		while (!parser.reached_end() && !parser.match("-->")) {
			parser.advance();
		}
		parser.skip_whitespace_or_newline();
	}
}

static XMLNode parse_tag(Parser & parser) {
	if (parser.reached_end()) return { };

	parser.expect('<');

	XMLNode node = { };
	node.location         = parser.location;
	node.is_question_mark = parser.match('?');

	// Parse node tag
	node.tag.start = parser.cur;
	while (!parser.reached_end() && !is_whitespace(*parser.cur) && *parser.cur != '>') parser.advance();
	node.tag.end = parser.cur;

	if (node.tag.length() == 0) {
		ERROR(parser.location, "Empty open tag!\n");
	} else if (node.tag.start[0] == '/') {
		ERROR(parser.location, "Unexpected closing tag '%.*s', expected open tag!\n", unsigned(node.tag.length()), node.tag.start);
	}

	parser_skip(parser);

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
		parser_skip(parser);

		node.attributes.push_back(attribute);

		// Check if this is an inline tag (i.e. <tag/> or <?tag?>), if so return
		if (parser.match('/') || (node.is_question_mark && parser.match('?'))) {
			parser.expect('>');
			return node;
		}
	}

	parser_skip(parser);

	// Parse children
	while (!parser.match("</")) {
		node.children.push_back(parse_tag(parser));
		parser_skip(parser);
	}

	int i = 0;
	while (!parser.reached_end() && !parser.match('>')) {
		if (*parser.cur != node.tag.start[i++]) {
			ERROR(parser.location, "Non matching closing tag for '%.*s'!\n", unsigned(node.tag.length()), node.tag.start);
		}
		parser.advance();
	}

	return node;
}

static XMLNode parse_xml(Parser & parser) {
	XMLNode node;

	do {
		parser_skip(parser);
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

using ShapeGroupMap = HashMap<StringView, ShapeGroup,     StringViewHash>;
using SerializedMap = HashMap<StringView, Serialized,     StringViewHash>;
using MaterialMap   = HashMap<StringView, MaterialHandle, StringViewHash>;
using TextureMap    = HashMap<StringView, TextureHandle,  StringViewHash>;

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
			const StringView & filename_rel = reflectance->get_child_value<StringView>("filename");
			const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

			texture = scene.asset_manager.add_texture(filename_abs);

			const XMLNode * scale = reflectance->find_child_by_name("scale");
			if (scale) {
				rgb = scale->get_optional_attribute("value", Vector3(1.0f));
			}

			delete [] filename_abs;
			return;
		} else if (reflectance->tag == "ref") {
			const StringView & texture_name = reflectance->get_child_value<StringView>("id");
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
			Matrix4 world = matrix->get_attribute_value<Matrix4>("value");
			Matrix4::decompose(world, position, rotation, scale, forward);
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
			return matrix->get_attribute_value<Matrix4>("value");
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
			material.emission = emitter->get_child_value<Vector3>("radiance");

			return scene.asset_manager.add_material(material);
		}

		// Check if an existing Material is referenced
		const XMLNode * ref = node->find_child("ref");
		if (ref) {
			const StringView & material_name = ref->get_attribute_value<StringView>("id");

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

	const XMLNode * inner_bsdf = bsdf;
	StringView inner_bsdf_type = inner_bsdf->get_attribute_value<StringView>("type");

	// Keep peeling back nested BSDFs, we only care about the innermost one
	while (
		inner_bsdf_type == "twosided" ||
		inner_bsdf_type == "mask" ||
		inner_bsdf_type == "bumpmap" ||
		inner_bsdf_type == "coating"
	) {
		inner_bsdf      = inner_bsdf->find_child("bsdf");
		inner_bsdf_type = inner_bsdf->get_attribute_value<StringView>("type");

		if (name == nullptr) {
			name = inner_bsdf->find_attribute("id");
		}
	}

	if (name) {
		material.name = name->value.c_str();
	} else {
		material.name = "Material";
	}

	if (inner_bsdf_type == "diffuse") {
		material.type = Material::Type::DIFFUSE;

		parse_rgb_or_texture(inner_bsdf, "reflectance", texture_map, path, scene, material.diffuse, material.texture_id);
	} else if (inner_bsdf_type == "conductor") {
		material.type = Material::Type::GLOSSY;

		parse_rgb_or_texture(inner_bsdf, "specularReflectance", texture_map, path, scene, material.diffuse, material.texture_id);

		material.linear_roughness = 0.0f;
		material.eta              = inner_bsdf->get_child_value_optional("eta", Vector3(1.33f));
		material.k                = inner_bsdf->get_child_value_optional("k",   Vector3(1.0f));
	} else if (inner_bsdf_type == "roughconductor" || inner_bsdf_type == "roughdiffuse") {
		material.type = Material::Type::GLOSSY;

		parse_rgb_or_texture(inner_bsdf, "specularReflectance", texture_map, path, scene, material.diffuse, material.texture_id);

		material.linear_roughness = inner_bsdf->get_child_value_optional("alpha", 0.5f);
		material.eta              = inner_bsdf->get_child_value_optional("eta",   Vector3(1.33f));
		material.k                = inner_bsdf->get_child_value_optional("k",     Vector3(1.0f));
	} else if (inner_bsdf_type == "plastic" || inner_bsdf_type == "roughplastic") {
		material.type = Material::Type::GLOSSY;

		parse_rgb_or_texture(inner_bsdf, "diffuseReflectance", texture_map, path, scene, material.diffuse, material.texture_id);

		float int_ior = inner_bsdf->get_child_value_optional("intIOR", 1.33f);
		float ext_ior = inner_bsdf->get_child_value_optional("extIOR", 1.0f);

		material.linear_roughness = inner_bsdf->get_child_value_optional("alpha", 0.5f);
		material.eta              = Vector3(int_ior / ext_ior);
		material.k                = Vector3(5.0f);

		const XMLNode * nonlinear = inner_bsdf->find_child_by_name("nonlinear");
		if (nonlinear && nonlinear->get_attribute_value<bool>("value")) {
			material.linear_roughness = sqrtf(material.linear_roughness);
		}
	} else if (inner_bsdf_type == "thindielectric" || inner_bsdf_type == "dielectric" || inner_bsdf_type == "roughdielectric") {
		float int_ior = inner_bsdf->get_child_value_optional("intIOR", 1.33f);
		float ext_ior = inner_bsdf->get_child_value_optional("extIOR", 1.0f);

		material.type = Material::Type::DIELECTRIC;
		material.transmittance       = Vector3(1.0f);
		material.index_of_refraction = int_ior / ext_ior;

		const XMLNode * medium = node->find_child("medium");
		if (medium) {
			Vector3 sigma_s = medium->get_child_value_optional("sigmaS", Vector3(0.0f, 0.0f, 0.0f));
			Vector3 sigma_a = medium->get_child_value<Vector3>("sigmaA");

			material.transmittance = Vector3(
				expf(-(sigma_a.x + sigma_s.x)),
				expf(-(sigma_a.y + sigma_s.y)),
				expf(-(sigma_a.z + sigma_s.z))
			);
		}
	} else if (inner_bsdf_type == "difftrans") {
		material.type = Material::Type::DIFFUSE;

		parse_rgb_or_texture(inner_bsdf, "transmittance", texture_map, path, scene, material.diffuse, material.texture_id);
	} else {
		WARNING(inner_bsdf->location, "WARNING: BSDF type '%.*s' not supported!\n", unsigned(inner_bsdf_type.length()), inner_bsdf_type.start);

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

		Parser parser = { };
		parser.init(deserialized, deserialized + deserialized_length, filename);

		// Read flags field
		uint32_t flags = parser.parse_binary<uint32_t>();
		bool flag_has_normals      = flags & 0x0001;
		bool flag_has_tex_coords   = flags & 0x0002;
		bool flag_has_colours      = flags & 0x0008;
		bool flag_use_face_normals = flags & 0x0010;
		bool flag_single_precision = flags & 0x1000;
		bool flag_double_precision = flags & 0x2000;

		// Read null terminated name
		mesh_names[i] = parser.parse_c_str().c_str();

		// Read number of vertices and triangles1
		uint64_t num_vertices  = parser.parse_binary<uint64_t>();
		uint64_t num_triangles = parser.parse_binary<uint64_t>();

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

		const char * vertex_positions = parser.cur;
		parser.advance(num_vertices * 3 * element_size);

		const char * vertex_normals = parser.cur;
		if (flag_has_normals) {
			parser.advance(num_vertices * 3 * element_size);
		}

		const char * vertex_tex_coords = parser.cur;
		if (flag_has_tex_coords) {
			parser.advance(num_vertices * 2 * element_size);
		}

		// Vertex colours, not used
		if (flag_has_colours) {
			parser.advance(num_vertices * 3 * element_size);
		}

		const char * indices = parser.cur;

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

			triangles[t].init();
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

static void parse_hair(const XMLNode * node, const char * filename, Triangle *& triangles, int & triangle_count, const Matrix4 & transform, float radius) {
	int          file_length;
	const char * file = Util::file_read(filename, file_length);

	Parser parser = { };
	parser.init(file, file + file_length, filename);

	Array<Array<Vector3>> hairs;
	Array<Vector3>        strand;

	triangle_count = 0;

	if (parser.match("BINARY_HAIR")) { // Binary format
		unsigned num_vertices = parser.parse_binary<unsigned>();

		while (parser.cur < parser.end) {
			float x = parser.parse_binary<float>();
			if (isinf(x)) { // +INF marks beginning of new hair strand
				triangle_count += strand.size() - 1;
				hairs.push_back(strand);
				strand.clear();
			} else {
				float y = parser.parse_binary<float>();
				float z = parser.parse_binary<float>();
				strand.emplace_back(x, y, z);
			}
		}
	} else { // ASCII format
		while (parser.cur < parser.end) {
			if (is_newline(*parser.cur)) { // Empty line marks beginning of new hair strand
				triangle_count += strand.size() - 1;
				hairs.push_back(strand);
				strand.clear();
			} else {
				float x = parser.parse_float();
				float y = parser.parse_float();
				float z = parser.parse_float();
				strand.emplace_back(x, y, z);
			}
			parser.parse_newline();
		}
	}

	if (strand.size() > 0) {
		triangle_count += strand.size() - 1;
		hairs.push_back(strand);
	}

	delete [] file;

	triangles = new Triangle[triangle_count];
	int current_triangle = 0;

	for (int h = 0; h < hairs.size(); h++) {
		Array<Vector3> & strand = hairs[h];

		for (int s = 0; s < strand.size(); s++) {
			strand[s] = Matrix4::transform_position(transform, strand[s]);
		}

		for (int s = 0; s < strand.size() - 1; s++) {
			Vector3 direction = Vector3::normalize(strand[s+1] - strand[s]);

			triangles[current_triangle].position_0  = strand[s] - radius * Math::orthogonal(direction);
			triangles[current_triangle].position_1  = strand[s] + radius * Math::orthogonal(direction);
			triangles[current_triangle].position_2  = strand[s+1];
			triangles[current_triangle].normal_0    = Vector3(0.0f);
			triangles[current_triangle].normal_1    = Vector3(0.0f);
			triangles[current_triangle].normal_2    = Vector3(0.0f);
			triangles[current_triangle].tex_coord_0 = Vector2(0.0f, 0.0f);
			triangles[current_triangle].tex_coord_1 = Vector2(1.0f, 0.0f);
			triangles[current_triangle].tex_coord_2 = Vector2(0.0f, 1.0f);
			triangles[current_triangle].init();

			current_triangle++;
		}
	}
}

static MeshDataHandle parse_shape(const XMLNode * node, Scene & scene, SerializedMap & serialized_map, const char * path, const char *& name) {
	StringView type = node->get_attribute_value<StringView>("type");

	if (type == "obj" || type == "ply") {
		const StringView & filename_rel = node->get_child_value<StringView>("filename");
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		MeshDataHandle mesh_data_handle = scene.asset_manager.add_mesh_data(filename_abs);
		delete [] filename_abs;

		name = filename_rel.c_str();

		return mesh_data_handle;

	} else if (type == "rectangle" || type == "cube" || type == "disk" || type == "cylinder" || type == "sphere" || type == "hair") {
		Matrix4 world = parse_transform_matrix(node);

		Triangle * triangles = nullptr;
		int        triangle_count = 0;

		if (type == "rectangle") {
			Geometry::rectangle(triangles, triangle_count, world);
		} else if (type == "cube") {
			Geometry::cube(triangles, triangle_count, world);
		} else if (type == "disk") {
			Geometry::disk(triangles, triangle_count, world);
		} else if (type == "cylinder") {
			Vector3 p0     = node->get_child_value_optional("p0", Vector3(0.0f, 0.0f, 0.0f));
			Vector3 p1     = node->get_child_value_optional("p1", Vector3(0.0f, 0.0f, 1.0f));
			float   radius = node->get_child_value_optional("radius", 1.0f);

			Geometry::cylinder(triangles, triangle_count, world, p0, p1, radius);
		} else if (type == "sphere") {
			float   radius = node->get_child_value_optional("radius", 1.0f);
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
		} else if (type == "hair") {
			const StringView & filename_rel = node->get_child_value<StringView>("filename");
			const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

			float radius = node->get_child_value_optional("radius", 0.0025f);

			parse_hair(node, filename_abs, triangles, triangle_count, world, radius);
		} else {
			abort(); // Unreachable
		}

		name = type.c_str();

		return scene.asset_manager.add_mesh_data(triangles, triangle_count);
	} else if (type == "serialized") {
		const StringView & filename_rel = node->get_child_value<StringView>("filename");

		Serialized serialized;
		bool found = serialized_map.try_get(filename_rel, serialized);
		if (!found) {
			const char * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

			serialized = parse_serialized(node, filename_abs, scene);
			serialized_map[filename_rel] = serialized;

			delete [] filename_abs;
		}

		int shape_index = node->get_child_value_optional("shapeIndex", 0);

		name = serialized.mesh_names[shape_index];
		return serialized.mesh_data_handles[shape_index];
	} else {
		WARNING(node->location, "WARNING: Shape type '%.*s' not supported!\n", unsigned(type.length()), type.start);
		return MeshDataHandle { INVALID };
	}
}

static void walk_xml_tree(const XMLNode * node, Scene & scene, ShapeGroupMap & shape_group_map, SerializedMap & serialized_map, MaterialMap & material_map, TextureMap & texture_map, const char * path) {
	if (node->tag == "bsdf") {
		MaterialHandle   material_handle = parse_material(node, scene, material_map, texture_map, path);
		const Material & material = scene.get_material(material_handle);

		StringView str = { material.name, material.name + strlen(material.name) };
		material_map[str] = material_handle;
	} else if (node->tag == "texture") {
		const StringView & filename_rel = node->get_child_value<StringView>("filename");
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
		StringView type = node->get_attribute_value<StringView>("type");
		if (type == "shapegroup") {
			const XMLNode * shape = node->find_child("shape");

			const char * name = nullptr;
			MeshDataHandle mesh_data_handle = parse_shape(shape, scene, serialized_map, path, name);
			MaterialHandle material_handle  = parse_material(shape, scene, material_map, texture_map, path);

			const StringView & id = node->get_attribute_value<StringView>("id");
			shape_group_map[id] = { mesh_data_handle, material_handle };
		} else if (type == "instance") {
			StringView id = node->get_child_value<StringView>("ref", "id");

			ShapeGroup shape_group;
			if (shape_group_map.try_get(id, shape_group) && shape_group.mesh_data_handle.handle != INVALID) {
				Mesh & mesh = scene.add_mesh(id.c_str(), shape_group.mesh_data_handle, shape_group.material_handle);
				parse_transform(node, &mesh.position, &mesh.rotation, &mesh.scale);
			}
		} else {
			char const * name = nullptr;
			MeshDataHandle mesh_data_handle = parse_shape(node, scene, serialized_map, path, name);
			MaterialHandle material_handle  = parse_material(node, scene, material_map, texture_map, path);

			if (mesh_data_handle.handle != INVALID) {
				Mesh & mesh = scene.add_mesh(name, mesh_data_handle, material_handle);

				// Do not apply transform to primitive shapes, since they have the transform baked into their vertices
				if (type == "obj" || type == "serialized") {
					parse_transform(node, &mesh.position, &mesh.rotation, &mesh.scale);
				}
			}
		}
	} else if (node->tag == "sensor") {
		const StringView & camera_type = node->get_attribute_value<StringView>("type");

		if (camera_type == "perspective" || camera_type == "perspective_rdist" || camera_type == "thinlens") {
			float fov = node->get_child_value_optional("fov", 110.0f);
			scene.camera.set_fov(Math::deg_to_rad(fov));
			scene.camera.aperture_radius = node->get_child_value_optional("aperatureRadius", 0.05f);
			scene.camera.focal_distance  = node->get_child_value_optional("focusDistance", 10.0f);

			float scale = 1.0f;
			parse_transform(node, &scene.camera.position, &scene.camera.rotation, &scale, Vector3(0.0f, 0.0f, -1.0f));

			if (scale < 0.0f) {
				scene.camera.rotation = Quaternion::conjugate(scene.camera.rotation);
			}
		} else {
			WARNING(node->location, "WARNING: Camera type '%.*s' not supported!\n", unsigned(camera_type.length()), camera_type.start);
		}
	} else if (node->tag == "include") {
		const StringView & filename_rel = node->get_attribute_value<StringView>("filename");
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

	Parser parser = { };
	parser.init(source, source + source_length, filename);

	XMLNode root = parse_xml(parser);

	ShapeGroupMap shape_group_map;
	SerializedMap serialized_map;
	MaterialMap   material_map;
	TextureMap    texture_map;
	char path[512];	Util::get_path(filename, path);
	walk_xml_tree(&root, scene, shape_group_map, serialized_map, material_map, texture_map, path);

	delete [] source;
}
