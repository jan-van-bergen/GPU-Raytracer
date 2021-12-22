#include "XMLParser.h"

inline void parser_skip(Parser & parser) {
	parser.skip_whitespace_or_newline();

	while (parser.match("<!--")) {
		while (!parser.reached_end() && !parser.match("-->")) {
			parser.advance();
		}
		parser.skip_whitespace_or_newline();
	}
}

inline XMLNode parse_tag(Parser & parser) {
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

XMLNode XMLParser::parse_root() {
	XMLNode node;

	do {
		parser_skip(parser);
		node = parse_tag(parser);
	} while (node.is_question_mark);

	return node;
}
