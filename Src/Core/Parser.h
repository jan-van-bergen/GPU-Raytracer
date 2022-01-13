#pragma once
#include <stdio.h>
#include <stdlib.h>

#include "IO.h"
#include "StringView.h"

#include "Util/Util.h"

#ifdef _MSC_VER
#define strcasecmp _stricmp
#define strncasecmp _strnicmp
#endif

#define TAB_WIDTH 4

#define WARNING(loc, msg, ...) \
	if (loc.file.length() > 0) { \
		IO::print("{}:{}:{}: " msg ""sv, loc.file, loc.line, loc.col, __VA_ARGS__); \
	} else { \
		IO::print(msg ""sv, __VA_ARGS__); \
	}

#define ERROR(loc, msg, ...) \
	WARNING(loc, msg, __VA_ARGS__); \
	abort();

inline bool is_digit(char c) {
	return c >= '0' && c <= '9';
}

inline bool is_whitespace(char c) {
	return c == ' ' || c == '\t';
}

inline bool is_newline(char c) {
	return c == '\r' || c == '\n';
}

struct SourceLocation {
	StringView file;
	int        line;
	int        col;

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

inline const char * char_to_str(char c) {
	// Based on: https://www.rapidtables.com/code/text/ascii-table.html
	static const char * table[] = {
		"\\0", "SOH", "STX", "ETX", "EOT", "ENQ", "ACK", "\\a", "\\b", "\\t", "\\n", "\\v", "\\f", "\\r", "SO", "SI",
		"DLE", "DC1", "DC2", "DC3", "DC4", "NAK", "SYN", "ETB", "CAN", "EM",  "SUB", "\\e", "FS",  "GS",  "RS", "US",
		" ",   "!",   "\"",  "#",   "$",   "%",   "&",   "'",   "(",   ")",   "*",   "+",   ",",   "-",   ".",   "/",
		"0",   "1",   "2",   "3",   "4",   "5",   "6",   "7",   "8",   "9",   ":",   ";",   "<",   "=",   ">",   "?",
		"@",   "A",   "B",   "C",   "D",   "E",   "F",   "G",   "H",   "I",   "J",   "K",   "L",   "M",   "N",   "O",
		"P",   "Q",   "R",   "S",   "T",   "U",   "V",   "W",   "X",   "Y",   "Z",   "[",   "\\",  "]",   "^",   "_",
		"`",   "a",   "b",   "c",   "d",   "e",   "f",   "g",   "h",   "i",   "j",   "k",   "l",   "m",   "n",   "o",
		"p",   "q",   "r",   "s",   "t",   "u",   "v",   "w",   "x",   "y",   "z",   "{",   "|",   "}",   "~",   "DEL",

		// Non-ascii bytes, print as hex
		"0x80", "0x81", "0x82", "0x83", "0x84", "0x85", "0x86", "0x87", "0x88", "0x89", "0x8a", "0x8b", "0x8c", "0x8d", "0x8e", "0x8f",
		"0x90", "0x91", "0x92", "0x93", "0x94", "0x95", "0x96", "0x97", "0x98", "0x99", "0x9a", "0x9b", "0x9c", "0x9d", "0x9e", "0x9f",
		"0xa0", "0xa1", "0xa2", "0xa3", "0xa4", "0xa5", "0xa6", "0xa7", "0xa8", "0xa9", "0xaa", "0xab", "0xac", "0xad", "0xae", "0xaf",
		"0xb0", "0xb1", "0xb2", "0xb3", "0xb4", "0xb5", "0xb6", "0xb7", "0xb8", "0xb9", "0xba", "0xbb", "0xbc", "0xbd", "0xbe", "0xbf",
		"0xc0", "0xc1", "0xc2", "0xc3", "0xc4", "0xc5", "0xc6", "0xc7", "0xc8", "0xc9", "0xca", "0xcb", "0xcc", "0xcd", "0xce", "0xcf",
		"0xd0", "0xd1", "0xd2", "0xd3", "0xd4", "0xd5", "0xd6", "0xd7", "0xd8", "0xd9", "0xda", "0xdb", "0xdc", "0xdd", "0xde", "0xdf",
		"0xe0", "0xe1", "0xe2", "0xe3", "0xe4", "0xe5", "0xe6", "0xe7", "0xe8", "0xe9", "0xea", "0xeb", "0xec", "0xed", "0xee", "0xef",
		"0xf0", "0xf1", "0xf2", "0xf3", "0xf4", "0xf5", "0xf6", "0xf7", "0xf8", "0xf9", "0xfa", "0xfb", "0xfc", "0xfd", "0xfe", "0xff"
	};
	return table[unsigned char(c)];
}

struct Parser {
	const char * cur   = nullptr;
	const char * start = nullptr;
	const char * end   = nullptr;

	SourceLocation location = { };

	Parser(StringView data, StringView filename = { }) : Parser(data, SourceLocation { filename, 1, 0 }) { }

	Parser(StringView data, SourceLocation location) {
		this->cur   = data.start;
		this->start = data.start;
		this->end   = data.end;
		this->location = location;
	}

	bool reached_end() const {
		return cur >= end;
	}

	char advance(int n = 1) {
		if (cur + n > end) {
			ERROR(location, "Unexpected end of file!\n");
		}

		char c = *cur;

		for (int i = 0; i < n; i++) {
			location.advance(*cur);
			cur++;
		}

		return c;
	}

	void seek(size_t offset) {
		cur = start + offset;
	}

	void skip_whitespace() {
		while (cur < end && is_whitespace(*cur)) advance();
	}

	void skip_whitespace_or_newline() {
		while (cur < end && (is_whitespace(*cur) || is_newline(*cur))) advance();
	}

	char peek(int offset = 0) {
		if (cur + offset < end) {
			return cur[offset];
		}
		ERROR(location, "Unexpected end of file!\n");
	}

	bool match(char target) {
		if (cur < end && *cur == target) {
			advance();
			return true;
		}
		return false;
	}

	template<int N>
	bool match(const char (& target)[N]) {
		if (cur + N - 1 <= end && strncmp(cur, target, N - 1) == 0) {
			for (int i = 0; i < N - 1; i++) {
				advance();
			}
			return true;
		}
		return false;
	}

	template<int N>
	bool match_any_case(const char (& target)[N]) {
		if (cur + N - 1 <= end && strncasecmp(cur, target, N - 1) == 0) {
			for (int i = 0; i < N - 1; i++) {
				advance();
			}
			return true;
		}
		return false;
	}

	void expect(char expected) {
		if (reached_end()) {
			ERROR(location, "Unexpected end of file, expected '{}'!\n", char_to_str(expected));
		}
		if (*cur != expected) {
			ERROR(location, "Unexpected char '{}', expected '{}'!\n", char_to_str(*cur), char_to_str(expected))
		}
		advance();
	}

	template<int N>
	void expect(const char (& target)[N]) {
		for (int i = 0; i < N - 1; i++) {
			expect(target[i]);
		}
	}

	float parse_float() {
		if (match_any_case("nan")) {
			return NAN;
		}

		bool sign = false;
		if (match('-')) {
			 sign = true;
		} else if (match('+')) {
			 sign = false;
		}
		skip_whitespace();

		if (match_any_case("inf") || match_any_case("infinity")) {
			return sign ? -INFINITY : INFINITY;
		}

		double value = 0.0;

		bool has_integer_part    = false;
		bool has_fractional_part = false;

		// Parse integer part
		if (is_digit(*cur)) {
			value = parse_int();
			has_integer_part = true;
		}

		// Parse fractional part
		if (match('.')) {
			static constexpr double DIGIT_LUT[] = { 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001 };

			int digit = 0;
			while (is_digit(*cur)) {
				double p;
				if (digit < Util::array_count(DIGIT_LUT)) {
					p = DIGIT_LUT[digit];
				} else {
					p = pow(0.1, digit);
				}
				value += double(*cur - '0') * p;

				digit++;
				advance();
			}

			has_fractional_part = digit > 0;
		}

		if (!has_integer_part && !has_fractional_part) {
			ERROR(location, "Expected float, got '{}'", char_to_str(*cur));
		}

		// Parse exponent
		if (match('e') || match('E')) {
			int exponent = parse_int();
			value = value * pow(10.0, exponent);
		}

		return sign ? -value : value;
	}

	int parse_int() {
		bool sign = match('-');
		if (!sign) {
			match('+');
		}

		if (!is_digit(*cur)) {
			ERROR(location, "Expected integer digit, got '{}'", char_to_str(*cur));
		}

		int value = 0;
		while (is_digit(*cur)) {
			value *= 10;
			value += *cur - '0';
			advance();
		}

		return sign ? -value : value;
	}

	StringView parse_identifier() {
		skip_whitespace();

		const char * start = cur;
		while (!reached_end() && !is_whitespace(*cur) && !is_newline(*cur)) {
			advance();
		}

		return StringView { start, cur };
	}

	void parse_newline() {
		match('\r');
		expect('\n');
	}

	StringView parse_c_str() {
		const char * start = cur;
		while (*cur) {
			advance();
		}
		advance();
		return { start, cur - 1 };
	}

	template<typename T>
	T parse_binary() {
		const char * start = cur;
		advance(sizeof(T));

		T result;
		memcpy(&result, start, sizeof(T));
		return result;
	}
};
