#include "Format.h"

#include "Parser.h"

Format::Spec Format::parse_fmt(StringView fmt) const {
	Parser parser = { };
	parser.init(fmt);

	Spec spec = { };
	spec.fmt_end = fmt.end;
	spec.restart = fmt.end;

	while (!parser.reached_end()) {
		if (parser.match('{')) {
			if (parser.match('{')) continue; // A double {{ means escape

			spec.fmt_end = parser.cur - 1;

			if (parser.match(':')) {
				if (parser.match('<')) {
					spec.align = '<';
				} else if (parser.match('>')) {
					spec.align = '>';
				} else if (parser.match('^')) {
					spec.align = '^';
				} else if (parser.peek(1) == '<' || parser.peek(1) == '>' || parser.peek(1) == '^') {
					spec.fill  = parser.advance();
					spec.align = parser.advance();
				}

				if (parser.match('+')) {
					spec.sign = '+';
				} else if (parser.match('-')) {
					spec.sign = '-';
				} else if (parser.match(' ')) {
					spec.sign = ' ';
				}

				if (parser.match('#')) {
					spec.alternate = true;
				}
				if (parser.match('0')) {
					spec.zero_pad = true;
				}

				if (is_digit(*parser.cur)) {
					spec.width = parser.parse_int();
				}

				if (parser.match('.')) {
					spec.precision = parser.parse_int();
				}

				if (*parser.cur == 's' || *parser.cur == 'b' || *parser.cur == 'B' || *parser.cur == 'c' ||
					*parser.cur == 'd' || *parser.cur == 'o' || *parser.cur == 'x' || *parser.cur == 'X') {
					spec.type = parser.advance();
				}
			}

			parser.expect('}');
			spec.restart = parser.cur;

			break;
		} else {
			parser.advance();
		}
	}

	return spec;

}
