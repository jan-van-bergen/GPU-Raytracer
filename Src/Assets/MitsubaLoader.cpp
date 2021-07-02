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

static bool is_whitespace(char c) {
	return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static void skip_whitespace(const char *& cur, const char * end) {
	while (cur < end && is_whitespace(*cur)) cur++;
}

static bool match(const char *& cur, const char * end, char target) {
	if (cur < end && *cur == target) {
		cur++;
		return true;
	}

	return false;
}

static void expect(const char *& cur, const char * end, char expected) {
	if (cur >= end || *cur != expected) {
		if (cur < end) printf("Unexpected char '%c', expected '%c'!\n", *cur, expected);
		abort();
	}
	cur++;
}

template<typename T>
static float parse_number(const char *& cur, const char * end) {
	float T; 
	std::from_chars_result result = std::from_chars(cur, end, T);

	if ((bool)result.ec) abort();

	cur = result.ptr;
	return T;
}

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

	int value_as_int() const {
		const char * cur = value.start;
		return parse_number<int>(cur, value.end);
	}
	
	float value_as_float() const {
		const char * cur = value.start;
		return parse_number<float>(cur, value.end);
	}

	bool value_as_bool() const {
		if (value == "true")  return true;
		if (value == "false") return false;
		abort();
	}

	Vector3 value_as_vector3() const {
		const char * cur = value.start;

		int i = 0;
		Vector3 v;
		while (true) {
			v[i] = parse_number<float>(cur, value.end);

			if (i++ == 2) break;
			
			expect(cur, value.end, ',');
			match(cur, value.end, ' '); // optional
		}

		return v;
	}

	Matrix4 value_as_matrix4() const {
		const char * cur = value.start;

		int i = 0;
		Matrix4 m;
		while (true) {
			m.cells[i] = parse_number<float>(cur, value.end);

			if (i++ == 15) break;
			
			expect(cur, value.end, ' ');
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
};

static XMLNode parse_tag(const char *& cur, const char * end) {
	if (cur >= end) return { };

	expect(cur, end, '<');

	XMLNode node = { };
	node.is_question_mark = match(cur, end, '?');

	// Parse node tag
	node.tag.start = cur;
	while (cur < end && !is_whitespace(*cur)) cur++;
	node.tag.end = cur;

	skip_whitespace(cur, end);

	// Parse attributes
	while (cur < end && !match(cur, end, '>')) {
		XMLAttribute attribute = { };
		
		// Parse attribute name
		attribute.name.start = cur;
		while (cur < end && *cur != '=') cur++;
		attribute.name.end = cur;

		expect(cur, end, '=');
		expect(cur, end, '"');

		// Parse attribute value
		attribute.value.start = cur;
		while (cur < end && *cur != '"') cur++;
		attribute.value.end = cur;

		expect(cur, end, '"');
		skip_whitespace(cur, end);

		node.attributes.push_back(attribute);

		// Check if this is an inline tag (i.e. <tag/> or <?tag?>), if so return
		if (match(cur, end, '/') || (node.is_question_mark && match(cur, end, '?'))) {
			expect(cur, end, '>');
			return node;
		}
	}

	skip_whitespace(cur, end);

	// Parse children
	do {
		node.children.push_back(parse_tag(cur, end));
		skip_whitespace(cur, end);
	} while (!(cur + 1 < end && cur[0] == '<' && cur[1] == '/'));

	expect(cur, end, '<');
	expect(cur, end, '/');

	int i = 0;
	while (cur < end && !match(cur, end, '>')) {
		if (*cur != node.tag.start[i++]) {
			const char * str = node.tag.c_str();
			printf("ERROR: Non matching closing tag for '%s'!\n", str);
			delete [] str;
			abort();
		}
		cur++;
	}

	return node;
}

static XMLNode parse_xml(const char *& cur, const char * end) {
	XMLNode node;

	do {
		skip_whitespace(cur, end);
		node = parse_tag(cur, end);
	} while (node.is_question_mark);

	return node;
}

using MaterialMap = std::unordered_map<std::string, MaterialHandle>;

static const char * get_absolute_filename(const char * path, int len_path, const char * filename, int len_filename) {
	char * filename_abs = new char[len_path + len_filename + 1];

	memcpy(filename_abs,            path,     len_path);
	memcpy(filename_abs + len_path, filename, len_filename);
	filename_abs[len_path + len_filename] = '\0';

	return filename_abs;
}

static void find_rgb_or_texture(const XMLNode * node, const char * name, const char * path, Scene & scene, Vector3 & rgb, TextureHandle & texture) {
	const XMLNode * reflectance = node->find_child_by_name(name);
	if (reflectance == nullptr) {
		rgb = Vector3(1.0f);
	} else if (reflectance->tag == "rgb") {
		rgb = reflectance->find_attribute("value")->value_as_vector3();
	} else if (reflectance->tag == "texture") {
		const StringView & filename_rel = reflectance->find_child_by_name("filename")->find_attribute("value")->value;
		const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

		texture = scene.asset_manager.add_texture(filename_abs);

		const XMLNode * scale = reflectance->find_child_by_name("scale");
		if (scale) {
			rgb = scale->find_attribute("value")->value_as_vector3();
		}

		delete [] filename_abs;	
	} else {
		abort();
	}
}

static Matrix4 find_transform(const XMLNode * node) {
	const XMLNode * transform = node->find_child("transform");
	if (transform) {
		return transform->find_child("matrix")->find_attribute("value")->value_as_matrix4();
	} else {
		return Matrix4();
	}
}

static void decompose_matrix(const Matrix4 & matrix, Vector3 * position, Quaternion * rotation, float * scale) {
	if (position) *position = Vector3(matrix(0, 3), matrix(1, 3), matrix(2, 3));
	
	if (rotation) *rotation = Quaternion::look_rotation(
		Matrix4::transform_direction(matrix, Vector3(0.0f, 0.0f, 1.0f)),
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

static float find_optional_float(const XMLNode * node, const char * name, float default_value) {
	const XMLNode * child = node->find_child_by_name(name);
	if (child) {
		const XMLAttribute * attribute = child->find_attribute("value");
		if (attribute) {
			return attribute->value_as_float();
		}
	}

	return default_value;
}

static MaterialHandle find_material(const XMLNode * node, Scene & scene, MaterialMap & materials, const char * path) {
	// Check if an existing Material is referenced
	const XMLNode * ref = node->find_child("ref");
	if (ref) {
		const StringView & material_name = ref->find_attribute("id")->value;

		auto material_id = materials.find(std::string(material_name.start, material_name.end));
		if (material_id == materials.end()) {
			const char * str = material_name.c_str();
			printf("Invalid material Ref '%s'!\n", str);
			delete [] str;

			return MaterialHandle::get_default();
		}

		return material_id->second;
	}
	
	Material material = { };

	// Check if there is an emitter defined
	const XMLNode * emitter = node->find_child("emitter");
	if (emitter) {
		material.type = Material::Type::LIGHT;
		material.name = "emitter";
		material.emission = emitter->find_child_by_name("radiance")->find_attribute("value")->value_as_vector3();
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

		const XMLAttribute * type = bsdf->find_attribute("type");
		
		if (type->value == "bumpmap") {
			bsdf = bsdf->find_child("bsdf");
			type = bsdf->find_attribute("type");
		}

		const XMLAttribute * name = bsdf->find_attribute("id");

		const char * name_str = nullptr;
		if (name == nullptr) {
			name_str = "NO_NAME";
		} else {
			name_str = name->value.c_str();
		}
		material.name = name_str;

		if (type->value == "twosided") {
			const XMLNode      * inner_bsdf = bsdf->find_child("bsdf");
			const XMLAttribute * inner_bsdf_type = inner_bsdf->find_attribute("type");

			if (inner_bsdf_type->value == "diffuse") {
				material.type = Material::Type::DIFFUSE;
				
				find_rgb_or_texture(inner_bsdf, "reflectance", path, scene, material.diffuse, material.texture_id);
			} else if (inner_bsdf_type->value == "conductor") {
				material.type = Material::Type::GLOSSY;

				find_rgb_or_texture(inner_bsdf, "specularReflectance", path, scene, material.diffuse, material.texture_id);

				material.linear_roughness    = 0.0f;
				material.index_of_refraction = find_optional_float(bsdf, "eta", 1.0f);
			} else if (inner_bsdf_type->value == "roughconductor") {
				material.type = Material::Type::GLOSSY;

				find_rgb_or_texture(inner_bsdf, "specularReflectance", path, scene, material.diffuse, material.texture_id);

				material.linear_roughness    = find_optional_float(bsdf, "alpha", 0.5f);
				material.index_of_refraction = find_optional_float(bsdf, "eta",   1.0f);
			} else if (inner_bsdf_type->value == "roughplastic") {
				material.type = Material::Type::GLOSSY;
				
				find_rgb_or_texture(inner_bsdf, "diffuseReflectance", path, scene, material.diffuse, material.texture_id);

				material.linear_roughness    = find_optional_float(bsdf, "alpha" , 0.5f);
				material.index_of_refraction = find_optional_float(bsdf, "intIOR", 1.0f);
				
				const XMLNode * nonlinear = inner_bsdf->find_child_by_name("nonlinear");
				if (nonlinear && nonlinear->find_attribute("value")->value_as_bool()) {
					material.linear_roughness = sqrtf(material.linear_roughness);
				}
			} else {
				const char * str = inner_bsdf_type->value.c_str();
				printf("BSDF type '%s' not supported!\n", str);
				delete [] str;
			}
		} else if (type->value == "thindielectric" || type->value == "dielectric") {
			material.type = Material::Type::DIELECTRIC;
			material.transmittance = Vector3(1.0f);
			material.index_of_refraction = find_optional_float(bsdf, "intIOR", 1.0f);
		} else {
			const char * str = type->value.c_str();
			printf("Material type '%s' not supported!\n", str);
			delete [] str;
		}
	}

	return scene.asset_manager.add_material(material);
}

static void walk_xml_tree(const XMLNode & node, Scene & scene, MaterialMap & materials, const char * path) {
	if (node.tag == "bsdf") {
		MaterialHandle   material_id = find_material(&node, scene, materials, path);
		const Material & material = scene.asset_manager.get_material(material_id);

		materials[material.name] = material_id; 
	} else if (node.tag == "shape") {
		const XMLAttribute * type = node.find_attribute("type");
		if (type->value == "obj") {
			const StringView & filename_rel = node.find_child_by_name("filename")->find_attribute("value")->value;
			const char       * filename_abs = get_absolute_filename(path, strlen(path), filename_rel.start, filename_rel.length());

			MeshDataHandle mesh_data_id = scene.asset_manager.add_mesh_data(filename_abs);

			Mesh & mesh = scene.meshes.emplace_back();
			mesh.init(filename_abs, mesh_data_id, scene);
			mesh.material_id = find_material(&node, scene, materials, path);

			Matrix4 world = find_transform(&node);			
			decompose_matrix(world, &mesh.position, &mesh.rotation, &mesh.scale);
		} else if (type->value == "rectangle" || type->value == "cube" || type->value == "disk" || type->value == "sphere") {
			Matrix4 world = find_transform(&node);

			Triangle * triangles = nullptr;
			int        triangle_count = 0;

			if (type->value == "rectangle") {
				Geometry::rectangle(triangles, triangle_count, world);
			} else if (type->value == "cube") {
				Geometry::cube(triangles, triangle_count, world);
			} else if (type->value == "disk") {
				Geometry::disk(triangles, triangle_count, world);
			} else if (type->value == "sphere") {
				float radius = find_optional_float(&node, "radius", 1.0f);

				const XMLNode * xml_center = node.find_child_by_name("center");
				Vector3 center = Vector3(
					xml_center ? xml_center->find_attribute("x")->value_as_float() : 0.0f,
					xml_center ? xml_center->find_attribute("y")->value_as_float() : 0.0f,
					xml_center ? xml_center->find_attribute("z")->value_as_float() : 0.0f
				);

				world = world * Matrix4::create_translation(center) * Matrix4::create_scale(radius);

				Geometry::sphere(triangles, triangle_count, world);
			} else abort();

			MeshDataHandle mesh_data_id = scene.asset_manager.add_mesh_data(triangles, triangle_count);

			Mesh & mesh = scene.meshes.emplace_back();
			mesh.init("rectangle", mesh_data_id, scene);
			mesh.material_id = find_material(&node, scene, materials, path);
		} else {
			const char * str = type->value.c_str();
			printf("Shape type '%s' not supported!\n", str);
			delete [] str;
		}
	} else if (node.tag == "sensor") {
		const StringView & camera_type = node.find_attribute("type")->value;

		if (camera_type == "perspective") {
			float fov = node.find_child_by_name("fov")->find_attribute("value")->value_as_float();
			scene.camera.set_fov(Math::deg_to_rad(fov));

			Matrix4 world = find_transform(&node);
			decompose_matrix(world, &scene.camera.position, &scene.camera.rotation, nullptr);
		} else if (camera_type == "thinlens") {
			scene.camera.aperture_radius = node.find_child_by_name("aperatureRadius")->find_attribute("value")->value_as_float();
			scene.camera.focal_distance  = node.find_child_by_name("focusDistance")  ->find_attribute("value")->value_as_float();

			Matrix4 world = find_transform(&node);
			decompose_matrix(world, &scene.camera.position, &scene.camera.rotation, nullptr);
		} else {
			const char * str = camera_type.c_str();
			printf("Camera type '%s' not supported!\n", str);
			delete [] str;
		}
	} else for (int i = 0; i < node.children.size(); i++) {
		walk_xml_tree(node.children[i], scene, materials, path);
	}
}

void MitsubaLoader::load(const char * filename, Scene & scene) {
	const char * file = Util::file_read(filename);
	int          file_len = strlen(file);

	const char * cur = file;

	XMLNode root = parse_xml(cur, file + file_len);

	MaterialMap materials;
	char path[128];	Util::get_path(filename, path);
	walk_xml_tree(root, scene, materials, path);
}
