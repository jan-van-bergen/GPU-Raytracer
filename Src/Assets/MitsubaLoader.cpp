#include "MitsubaLoader.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>
#include <unordered_map>
#include <charconv>

#include "MeshData.h"

#include "BVH/Builders/BVHBuilder.h"
#include "BVH/Builders/CWBVHBuilder.h"

#include "Pathtracer/Scene.h"

#include "Util/Util.h"
#include "Util/Geometry.h"

#define TAB_WIDTH 4

#define PARSER_ERROR(loc, msg, ...) \
	printf("Mitsuba XML Parsing error at %s:%i:%i:\n" msg, loc.file, loc.line, loc.col, __VA_ARGS__); \
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
			PARSER_ERROR(location, "End of File!\n");
		}
		if (*cur != expected) {
			PARSER_ERROR(location, "Unexpected char '%c', expected '%c'!\n", *cur, expected)
		}
		advance();
	}

	template<typename T>
	T parse_number() {
		T number = { }; 
		std::from_chars_result result = std::from_chars(cur, end, number);

		if (bool(result.ec)) {
			PARSER_ERROR(location, "Failed to parse '%.*s' as a number!\n", unsigned(end - cur), cur);
		}

		cur = result.ptr;
		return number;
	}
};

struct StringView {
	const char * start;
	const char * end;

	inline char operator[](int index) const { return start[index]; }

	int length() const { return end - start; }

	char * c_str() const {
		int len = length();
		char * str = new char[len + 1];
		memcpy(str, start, len);
		str[len] = '\0';
		return str;
	}

	struct Hash {
		size_t operator()(const StringView & str) const {
			static constexpr int p = 31;
			static constexpr int m = 1e9 + 9;

			size_t hash = 0;
			size_t p_pow = 1;
			for (int i = 0; i < str.length(); i++) {
				hash = (hash + (str[i] - 'a' + 1) * p_pow) % m;
				p_pow = (p_pow * p) % m;
			}

			return hash;
		}
	};
};

template<int N>
static bool operator==(const StringView & a, const char (&b)[N]) {
	if (a.length() != N - 1) return false;

	for (int i = 0; i < N - 1; i++) {
		if (a[i] != b[i]) return false;
	}

	return true;
}

static bool operator==(const StringView & a, const char * b) {
	int length = a.length();

	for (int i = 0; i < length; i++) {
		if (a[i] != b[i] || b[i] == '\0') return false;
	}

	return true;
}

static bool operator==(const StringView & a, const StringView & b) {
	int length_a = a.length();
	int length_b = b.length();

	for (int i = 0; i < length_a; i++) {
		if (a[i] != b[i]) return false;
	}

	return true;
}

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
		PARSER_ERROR(location_of_value, "Unable to parse '%.*s' as boolean!\n", unsigned(value.end - value.start), value.start);
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
			PARSER_ERROR(parser.location, "An attribute must begin with either double or single quotes!\n");
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
			PARSER_ERROR(parser.location, "Non matching closing tag for '%.*s'!\n", unsigned(node.tag.end - node.tag.start), node.tag.start);
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

static const char * get_absolute_filename(const char * path, int len_path, const char * filename, int len_filename) {
	char * filename_abs = new char[len_path + len_filename + 1];

	memcpy(filename_abs,            path,     len_path);
	memcpy(filename_abs + len_path, filename, len_filename);
	filename_abs[len_path + len_filename] = '\0';

	return filename_abs;
}

static void find_rgb_or_texture(const XMLNode * node, const char * name, const char * path, Scene & scene, Vector3 & rgb, TextureHandle & texture) {
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
		}
	}
	rgb = Vector3(1.0f);
}

static Matrix4 parse_transform(const XMLNode * node) {
	const XMLNode * transform = node->find_child("transform");
	if (transform) {
		const XMLNode * matrix = transform->find_child("matrix");
		if (matrix) {
			return matrix->find_attribute("value")->get_value<Matrix4>();
		}
		
		const XMLNode * lookat = transform->find_child("lookat");
		if (lookat) {
			Vector3 origin = lookat->get_optional_attribute("origin", Vector3(0.0f, 0.0f,  0.0f));
			Vector3 target = lookat->get_optional_attribute("target", Vector3(0.0f, 0.0f, -1.0f));
			Vector3 up     = lookat->get_optional_attribute("up",     Vector3(0.0f, 1.0f,  0.0f));

			Vector3 forward = Vector3::normalize(target - origin);

			return Matrix4::create_translation(origin) * Matrix4::create_rotation(Quaternion::look_rotation(forward, up));
		}

		Matrix4 world;
		
		const XMLNode * scale = transform->find_child("scale");
		if (scale) {
			world = Matrix4::create_scale(scale->get_optional_attribute("value", 1.0f)) * world;
		}
		
		const XMLNode * rotate = transform->find_child("rotate");
		if (rotate) {
			float x = rotate->get_optional_attribute("x", 0.0f);
			float y = rotate->get_optional_attribute("y", 0.0f);
			float z = rotate->get_optional_attribute("z", 0.0f);

			if (x == 0.0f && y == 0.0f && z == 0.0f) {
				printf("WARNING: Rotation without axis specified!\n");
			} else {
				float angle = rotate->get_optional_attribute("angle", 0.0f);
				world = Matrix4::create_rotation(Quaternion::axis_angle(Vector3(x, y, z), Math::deg_to_rad(angle))) * world;
			}
		}

		const XMLNode * translate = transform->find_child("translate");
		if (translate) {
			world = Matrix4::create_translation(Vector3(
				translate->get_optional_attribute("x", 0.0f),
				translate->get_optional_attribute("y", 0.0f),
				translate->get_optional_attribute("z", 0.0f)
			)) * world;
		}

		return world;
	} else {
		return Matrix4();
	}
}

static void decompose_matrix(const Matrix4 & matrix, Vector3 * position, Quaternion * rotation, float * scale, const Vector3 & forward = Vector3(0.0f, 0.0f, 1.0f)) {
	if (position) *position = Vector3(matrix(0, 3), matrix(1, 3), matrix(2, 3));
	
	if (rotation) *rotation = Quaternion::look_rotation(
		Matrix4::transform_direction(matrix, forward),
		Matrix4::transform_direction(matrix, Vector3(0.0f, 1.0f, 0.0f))
	);

	if (scale) {
		float scale_x = Vector3::length(Vector3(matrix(0, 0), matrix(0, 1), matrix(0, 2)));
		float scale_y = Vector3::length(Vector3(matrix(1, 0), matrix(1, 1), matrix(1, 2)));
		float scale_z = Vector3::length(Vector3(matrix(2, 0), matrix(2, 1), matrix(2, 2)));

		if (Math::approx_equal(scale_x, scale_y) && Math::approx_equal(scale_y, scale_z)) {
			*scale = scale_x;
		} else {
			puts("Warning: nonuniform scaling!");
			*scale = cbrt(scale_x * scale_y * scale_y);
		}
	}
}

using MaterialMap = std::unordered_map<StringView, MaterialHandle, StringView::Hash>;

static MaterialHandle find_material(const XMLNode * node, Scene & scene, MaterialMap & material_map, const char * path) {
	const XMLNode * ref     = node->find_child("ref");
	const XMLNode * emitter = node->find_child("emitter");

	// Check if an existing Material is referenced
	if (ref && emitter == nullptr) { // If both a ref and emitter are defined, prefer the emitter
		const StringView & material_name = ref->find_attribute("id")->value;

		MaterialMap::iterator material_id = material_map.find(material_name);
		if (material_id == material_map.end()) {
			printf("Invalid material Ref '%.*s'!\n", unsigned(material_name.end - material_name.start), material_name.start);
			
			return MaterialHandle::get_default();
		}

		return material_id->second;
	}
	
	Material material = { };

	// Check if there is an emitter defined
	if (emitter) {
		material.type = Material::Type::LIGHT;
		material.name = "emitter";
		material.emission = emitter->find_child_by_name("radiance")->find_attribute("value")->get_value<Vector3>();
	} else {
		// Otherwise, parse BSDF
		const XMLNode * bsdf;
		if (node->tag == "bsdf") {
			bsdf = node;
		} else {
			bsdf = node->find_child("bsdf");
			if (bsdf == nullptr) {
				printf("Unable to parse BSDF!\n");
				return MaterialHandle::get_default();
			}
		}

		const XMLAttribute * name = bsdf->find_attribute("id");
		
		const XMLNode      * inner_bsdf = bsdf;
		const XMLAttribute * inner_bsdf_type = inner_bsdf->find_attribute("type");

		// Keep peeling back nested BSDFs, we only care about the inner one
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
				
			find_rgb_or_texture(inner_bsdf, "reflectance", path, scene, material.diffuse, material.texture_id);
		} else if (inner_bsdf_type->value == "conductor") {
			material.type = Material::Type::GLOSSY;

			find_rgb_or_texture(inner_bsdf, "specularReflectance", path, scene, material.diffuse, material.texture_id);

			material.linear_roughness    = 0.0f;
			material.index_of_refraction = bsdf->get_optional_child_value("eta", 1.0f);
		} else if (inner_bsdf_type->value == "roughconductor" || inner_bsdf_type->value == "roughdiffuse") {
			material.type = Material::Type::GLOSSY;

			find_rgb_or_texture(inner_bsdf, "specularReflectance", path, scene, material.diffuse, material.texture_id);

			material.linear_roughness    = bsdf->get_optional_child_value("alpha", 0.5f);
			material.index_of_refraction = bsdf->get_optional_child_value("eta",   1.0f);
		} else if (inner_bsdf_type->value == "plastic" || inner_bsdf_type->value == "roughplastic") {
			material.type = Material::Type::GLOSSY;

			find_rgb_or_texture(inner_bsdf, "diffuseReflectance", path, scene, material.diffuse, material.texture_id);

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
		} else {
			printf("WARNING: BSDF type '%.*s' not supported!\n", unsigned(inner_bsdf_type->value.end - inner_bsdf_type->value.start), inner_bsdf_type->value.start);
			
			return MaterialHandle::get_default();
		}
	}

	return scene.asset_manager.add_material(material);
}

static void walk_xml_tree(const XMLNode & node, Scene & scene, MaterialMap & material_map, const char * path) {
	if (node.tag == "bsdf") {
		MaterialHandle   material_id = find_material(&node, scene, material_map, path);
		const Material & material = scene.asset_manager.get_material(material_id);

		StringView str = { material.name, material.name + strlen(material.name) };
		material_map[str] = material_id; 
	} else if (node.tag == "shape") {
		const XMLAttribute * type = node.find_attribute("type");
		if (type->value == "obj") {
			const StringView & filename_rel = node.find_child_by_name("filename")->find_attribute("value")->value;
			const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

			MeshDataHandle mesh_data_id = scene.asset_manager.add_mesh_data(filename_abs);

			Mesh & mesh = scene.meshes.emplace_back();
			mesh.init(filename_rel.c_str(), mesh_data_id, scene);
			mesh.material_id = find_material(&node, scene, material_map, path);

			Matrix4 world = parse_transform(&node);			
			decompose_matrix(world, &mesh.position, &mesh.rotation, &mesh.scale);
		} else if (type->value == "rectangle" || type->value == "cube" || type->value == "disk" || type->value == "sphere") {
			Matrix4 world = parse_transform(&node);

			Triangle * triangles = nullptr;
			int        triangle_count = 0;

			if (type->value == "rectangle") {
				Geometry::rectangle(triangles, triangle_count, world);
			} else if (type->value == "cube") {
				Geometry::cube(triangles, triangle_count, world);
			} else if (type->value == "disk") {
				Geometry::disk(triangles, triangle_count, world);
			} else if (type->value == "sphere") {
				float   radius = node.get_optional_child_value("radius", 1.0f);
				Vector3 center = Vector3(0.0f);

				const XMLNode * xml_center = node.find_child_by_name("center");
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

			MeshDataHandle mesh_data_id = scene.asset_manager.add_mesh_data(triangles, triangle_count);

			Mesh & mesh = scene.meshes.emplace_back();
			mesh.init(type->value.c_str(), mesh_data_id, scene);
			mesh.material_id = find_material(&node, scene, material_map, path);
		} else {
			printf("WARNING: Shape type '%.*s' not supported!\n", unsigned(type->value.end - type->value.start), type->value.start);
		}
	} else if (node.tag == "sensor") {
		const StringView & camera_type = node.find_attribute("type")->value;

		if (camera_type == "perspective" || camera_type == "perspective_rdist" || camera_type == "thinlens") {
			float fov = node.get_optional_child_value("fov", 110.0f);
			scene.camera.set_fov(Math::deg_to_rad(fov));
			scene.camera.aperture_radius = node.get_optional_child_value("aperatureRadius", 0.05f);
			scene.camera.focal_distance  = node.get_optional_child_value("focusDistance", 10.0f);

			Matrix4 world = parse_transform(&node);
			decompose_matrix(world, &scene.camera.position, &scene.camera.rotation, nullptr, Vector3(0.0f, 0.0f, -1.0f));
		} else {
			printf("WARNING: Camera type '%.*s' not supported!\n", unsigned(camera_type.end - camera_type.start), camera_type.start);
		}
	} else for (int i = 0; i < node.children.size(); i++) {
		walk_xml_tree(node.children[i], scene, material_map, path);
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

	MaterialMap materials;
	char path[512];	Util::get_path(filename, path);
	walk_xml_tree(root, scene, materials, path);

	delete [] source;
}
