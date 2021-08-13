#pragma once
#include <stdio.h>
#include <stdlib.h>

#define TAB_WIDTH 4

#define WARNING(loc, msg, ...) \
	printf("%s:%i:%i: " msg, loc.file, loc.line, loc.col, __VA_ARGS__);

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
			ERROR(location, "End of File!\n");
		}
		if (*cur != expected) {
			ERROR(location, "Unexpected char '%c', expected '%c'!\n", *cur, expected)
		}
		advance();
	}

	float parse_float() {
		bool sign = false;
		if (match('-')) {
			 sign = true;
		} else if (match('+')) {
			 sign = false;
		}
		skip_whitespace();

		float value = float(parse_int());

		if (match('.')) {
			static constexpr float DIGIT_LUT[] = { 0.1f, 0.01f, 0.001f, 0.0001f, 0.00001f, 0.000001f, 0.0000001f, 0.00000001f, 0.000000001f, 0.0000000001f, 0.00000000001f };

			int digit = 0;
			while (is_digit(*cur)) {
				value += (*cur - '0') * DIGIT_LUT[digit++];
				advance();
			}
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
};
