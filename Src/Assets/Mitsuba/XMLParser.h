#pragma once
#include "Math/Vector3.h"
#include "Math/Matrix4.h"

#include "Util/Array.h"
#include "Util/Parser.h"

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

		bool uses_comma = parser.match(',');
		parser.skip_whitespace();

		if (!parser.reached_end()) {
			v.y = parser.parse_float();

			if (uses_comma) parser.expect(',');
			parser.skip_whitespace();

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
	const XMLAttribute * get_attribute(Predicate predicate) const {
		for (int i = 0; i < attributes.size(); i++) {
			if (predicate(attributes[i])) {
				return &attributes[i];
			}
		}
		return nullptr;
	}

	const XMLAttribute * get_attribute(const char * name) const {
		return get_attribute([name](const XMLAttribute & attribute) { return attribute.name == name; });
	}

	template<typename T>
	T get_attribute_optional(const char * name, T default_value) const {
		const XMLAttribute * attribute = get_attribute([name](const XMLAttribute & attribute) { return attribute.name == name; });
		if (attribute) {
			return attribute->get_value<T>();
		} else {
			return default_value;
		}
	}

	template<typename T = StringView>
	T get_attribute_value(const char * name) const {
		const XMLAttribute * attribute = get_attribute(name);
		if (attribute) {
			return attribute->get_value<T>();
		} else {
			ERROR(location, "Node '%.*s' does not have an attribute with name '%s'!\n", unsigned(tag.length()), tag.start, name);
		}
	}

	template<typename Predicate>
	const XMLNode * get_child(Predicate predicate) const {
		for (int i = 0; i < children.size(); i++) {
			if (predicate(children[i])) {
				return &children[i];
			}
		}
		return nullptr;
	}

	const XMLNode * get_child_by_tag(const char * tag) const {
		return get_child([tag](const XMLNode & node) { return node.tag == tag; });
	}

	const XMLNode * get_child_by_name(const char * name) const {
		return get_child([name](const XMLNode & node) {
			const XMLAttribute * attr = node.get_attribute("name");
			if (attr == nullptr) return false;

			return attr->value == name;
		});
	}

	template<typename T = StringView>
	T get_child_value(const char * child_name) const {
		const XMLNode * child = get_child_by_name(child_name);
		if (child) {
			return child->get_attribute_value<T>("value");
		} else {
			ERROR(location, "Node '%.*s' does not have a child with name '%s'!\n", unsigned(tag.length()), tag.start, child_name);
		}
	}

	template<typename T = StringView>
	T get_child_value_optional(const char * name, T default_value) const {
		const XMLNode * child = get_child_by_name(name);
		if (child) {
			return child->get_attribute_optional("value", default_value);
		} else {
			return default_value;
		}
	}
};

struct XMLParser {
	int          source_length;
	const char * source;

	Parser parser;

	void init(const char * filename) {
		source = Util::file_read(filename, source_length);
		parser.init(source, source + source_length, filename);
	}

	void free() {
		delete [] source;
	}

	XMLNode parse_root();
};
