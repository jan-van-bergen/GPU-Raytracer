#include "XMLParser.h"

XMLNode XMLParser::parse_root() {
	XMLNode root = XMLNode(allocator);
	root.location = parser.location;

	while (!parser.reached_end()) {
		parser_skip_xml_whitespace(parser);
		root.children.push_back(parse_tag());
		parser_skip_xml_whitespace(parser);
	}

	return root;
}

XMLNode XMLParser::parse_tag() {
	XMLNode node = XMLNode(allocator);
	if (parser.reached_end()) {
		return node;
	}

	parser.expect('<');

	node.location         = parser.location;
	node.is_question_mark = parser.match('?');

	// Parse node tag
	node.tag.start = parser.cur;
	while (!parser.reached_end() && !is_whitespace(*parser.cur) && *parser.cur != '>') parser.advance();
	node.tag.end = parser.cur;

	if (node.tag.length() == 0) {
		ERROR(parser.location, "Empty open tag!\n");
	} else if (node.tag.start[0] == '/') {
		ERROR(parser.location, "Unexpected closing tag '{}', expected open tag!\n", node.tag);
	}

	parser_skip_xml_whitespace(parser);

	// Parse attributes
	while (!parser.reached_end() && !parser.match('>')) {
		XMLAttribute attribute = { };

		// Parse attribute name
		attribute.name.start = parser.cur;
		while (!parser.reached_end() && *parser.cur != '=') parser.advance();
		attribute.name.end = parser.cur;

		parser.expect('=');
		while (!parser.reached_end() && *parser.cur != '"' && *parser.cur != '\'') parser.advance();

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
		parser_skip_xml_whitespace(parser);

		node.attributes.push_back(attribute);

		// Check if this is an inline tag (i.e. <tag/> or <?tag?>), if so return
		if (parser.match('/') || (node.is_question_mark && parser.match('?'))) {
			parser.expect('>');
			return node;
		}
	}

	parser_skip_xml_whitespace(parser);

	// Parse children
	while (!parser.match("</")) {
		node.children.push_back(parse_tag());
		parser_skip_xml_whitespace(parser);
	}

	const char * closing_tag_start = parser.cur;
	while (!parser.reached_end() && !parser.match('>')) {
		parser.advance();
	}

	StringView closing_tag = { closing_tag_start, parser.cur - 1 };
	if (node.tag != closing_tag) {
		ERROR(parser.location, "Non matching closing tag '{}' for Node '{}'!\n", closing_tag, node.tag);
	}

	return node;
}
