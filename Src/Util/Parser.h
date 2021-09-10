#pragma once
#include <stdio.h>
#include <stdlib.h>

#include "StringView.h"

#define TAB_WIDTH 4

#define WARNING(loc, msg, ...) \
	if (loc.file) { \
		printf("%s:%i:%i: " msg, loc.file, loc.line, loc.col, __VA_ARGS__); \
	} else { \
		printf(msg, __VA_ARGS__); \
	}

#define ERROR(loc, msg, ...) \
	WARNING(loc, msg, __VA_ARGS__); \
	abort();

static bool is_digit(char c) {
	return c >= '0' && c <= '9';
}

static bool is_whitespace(char c) {
	return c == ' ' || c == '\t';
}

static bool is_newline(char c) {
	return c == '\r' || c == '\n';
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

struct Parser {
	const char * cur;
	const char * end;

	SourceLocation location;

	void init(const char * cur, const char * end, const char * filename = nullptr) {
		this->cur = cur;
		this->end = end;
		location.file = filename;
		location.line = 1;
		location.col  = 0;
	}

	void init(const char * cur, const char * end, SourceLocation location) {
		this->cur = cur;
		this->end = end;
		this->location = location;
	}

	bool reached_end() const {
		return cur >= end;
	}

	void advance(int n = 1) {
		if (cur + n > end) {
			ERROR(location, "Unexpected end of file!\n");
		}
		for (int i = 0; i < n; i++) {
			location.advance(*cur);
			cur++;
		}
	}

	void skip_whitespace() {
		while (cur < end && is_whitespace(*cur)) advance();
	}

	void skip_whitespace_or_newline() {
		while (cur < end && (is_whitespace(*cur) || is_newline(*cur))) advance();
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
		if (cur + N - 1 < end && strncmp(cur, target, N - 1) == 0) {
			for (int i = 0; i < N - 1; i++) {
				advance();
			}
			return true;
		}
		return false;
	}

	void expect(char expected) {
		if (reached_end()) {
			ERROR(location, "Unexpected end of file, expected '%c'!\n", expected);
		}
		if (*cur != expected) {
			ERROR(location, "Unexpected char '%c', expected '%c'!\n", *cur, expected)
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
		bool sign = false;
		if (match('-')) {
			 sign = true;
		} else if (match('+')) {
			 sign = false;
		}
		skip_whitespace();

		double value = 0.0;

		// Parse integer part
		if (is_digit(*cur)) {
			value = parse_int();
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
