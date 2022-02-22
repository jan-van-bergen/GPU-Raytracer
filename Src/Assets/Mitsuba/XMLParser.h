#pragma once
#include "Math/Vector3.h"
#include "Math/Matrix4.h"

#include "Core/Array.h"
#include "Core/Parser.h"
#include "Core/Allocators/LinearAllocator.h"

inline void parser_skip_xml_whitespace(Parser & parser) {
	parser.skip_whitespace_or_newline();

	while (parser.match("<!--")) {
		while (!parser.reached_end() && !parser.match("-->")) {
			parser.advance();
		}
		parser.skip_whitespace_or_newline();
	}
}

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
		Parser parser(value, location_of_value);
		return parser.parse_int();
	}

	template<>
	float get_value() const {
		Parser parser(value, location_of_value);
		return parser.parse_float();
	}

	template<>
	bool get_value() const {
		if (value == "true")  return true;
		if (value == "false") return false;
		ERROR(location_of_value, "Unable to parse '{}' as boolean!\n", value);
	}

	template<>
	Vector3 get_value() const {
		Parser parser(value, location_of_value);
		parser_skip_xml_whitespace(parser);

		Vector3 v;
		v.x = parser.parse_float();

		bool uses_comma = parser.match(',');
		parser_skip_xml_whitespace(parser);

		if (!parser.reached_end()) {
			v.y = parser.parse_float();

			if (uses_comma) parser.expect(',');
			parser_skip_xml_whitespace(parser);

			v.z = parser.parse_float();
		} else {
			v.y = v.x;
			v.z = v.x;
		}

		return v;
	}

	template<>
	Matrix4 get_value() const {
		Parser parser(value, location_of_value);

		Matrix4 m;
		for (int i = 0; i < 16; i++) {
			parser_skip_xml_whitespace(parser);
			m.cells[i] = parser.parse_float();
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

	XMLNode(Allocator * allocator) : attributes(allocator), children(allocator) { }

	DEFAULT_MOVEABLE(XMLNode);
	DEFAULT_COPYABLE(XMLNode);

	~XMLNode() { }

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
			ERROR(location, "Node '{}' does not have an attribute with name '{}'!\n", tag, name);
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
			ERROR(location, "Node '{}' does not have a child with name '{}'!\n", tag, child_name);
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
	Allocator * allocator = nullptr;

	String source;
	Parser parser;

	XMLParser(const String & filename, Allocator * allocator) : allocator(allocator), source(IO::file_read(filename, allocator)), parser(source.view(), filename.view()) { }

	XMLNode parse_root();

private:
	XMLNode parse_tag();
};
