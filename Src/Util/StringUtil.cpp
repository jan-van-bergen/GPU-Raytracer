#include "StringUtil.h"

#include "Core/Assertion.h"
#include "Core/StringView.h"

#include "Util.h"

StringView Util::get_directory(StringView filename) {
	const char * last_slash = find_last_after(filename, "/\\"_sv);

	if (last_slash != filename.end) {
		return StringView { filename.start, last_slash };
	} else {
		return "./"_sv;
	}
}

StringView Util::remove_directory(StringView filename) {
	const char * last_slash = find_last_after(filename, "/\\"_sv);

	if (last_slash != filename.end) {
		return StringView { last_slash, filename.end };
	} else {
		return filename;
	}
}

StringView Util::get_file_extension(StringView filename) {
	return StringView { find_last_after(filename, "."_sv), filename.end };
}

StringView Util::substr(StringView str, size_t offset, size_t len) {
	if (offset + len >= str.size()) {
		len = str.size() - offset;
	}
	return { str.data() + offset, str.data() + offset + len };
}

String Util::combine_stringviews(StringView a, StringView b, Allocator * allocator) {
	String filename_abs = String(a.size() + b.size(), allocator);
	memcpy(filename_abs.data(),            a.start, a.size());
	memcpy(filename_abs.data() + a.size(), b.start, b.size());
	filename_abs[a.size() + b.size()] = '\0';
	return filename_abs;
}

const char * Util::find_last_after(StringView haystack, StringView needles) {
	const char * cur        = haystack.data();
	const char * last_match = nullptr;

	while (cur < haystack.end) {
		for (int i = 0; i < needles.size(); i++) {
			if (*cur == needles[i]) {
				last_match = cur;
				break;
			}
		}
		cur++;
	}

	if (last_match) {
		return last_match + 1;
	} else {
		return haystack.end;
	}
}

const char * Util::strstr(StringView haystack, StringView needle) {
	if (needle.is_empty()) return haystack.end;

	const char * cur = haystack.data();
	while (cur <= haystack.end - needle.size()) {
		bool match = true;
		for (int i = 0; i < needle.size(); i++) {
			if (cur[i] != needle[i]) {
				match = false;
				break;
			}
		}
		if (match) {
			return cur;
		}
		cur++;
	}

	return haystack.end;
}

static constexpr char DIGITS[] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };

static int int64_to_string(int64_t value, int64_t base, char * buf) {
	if (base > Util::array_count(DIGITS)) {
		base = Util::array_count(DIGITS);
	}

	bool negative = value < 0;

	int offset = 0;
	do {
		int digit = abs(value % base);
		buf[offset++] = DIGITS[digit];
		value /= base;
	} while (value != 0);
	ASSERT(offset > 0);

	if (negative) {
		buf[offset++] = '-';
		value = -value;
	}

	Util::reverse(buf, offset);

	return offset;
}

static int uint64_to_string(uint64_t value, uint64_t base, char * buf) {
	if (base > Util::array_count(DIGITS)) {
		base = Util::array_count(DIGITS);
	}

	int offset = 0;
	do {
		int digit = value % base;
		buf[offset++] = DIGITS[digit];
		value /= base;
	} while (value != 0);
	ASSERT(offset > 0);

	Util::reverse(buf, offset);

	return offset;
}

String Util::to_string(bool value, Allocator * allocator) {
	return String(value ? "true"_sv : "false"_sv, allocator);
}

String Util::to_string(int64_t value, int64_t base, Allocator * allocator) {
	char buf[64] = { };
	int length = int64_to_string(value, base, buf);
	return String(buf, length, allocator);
}

String Util::to_string(uint64_t value, uint64_t base, Allocator * allocator) {
	char buf[64] = { };
	int length = uint64_to_string(value, base, buf);
	return String(buf, length, allocator);
}

String Util::to_string(double value, Allocator * allocator) {
	if (isinf(value)) {
		return String(value > 0.0f ? "inf"_sv : "-inf"_sv, allocator);
	}
	if (isnan(value)) {
		return String("nan"_sv, allocator);
	}

	// Based on: https://github.com/antongus/stm32tpl/blob/master/ftoa.c

	int integer_part = int(value);
	value = value - floor(value);

	static constexpr size_t MAX_PRECISION = 10;
	static constexpr double rounders[MAX_PRECISION + 1] = {
		0.5,
		0.05,
		0.005,
		0.0005,
		0.00005,
		0.000005,
		0.0000005,
		0.00000005,
		0.000000005,
		0.0000000005,
		0.00000000005
	};

	int precision = 6;
	value += rounders[precision];

	char buf[64] = { };
	int offset = int64_to_string(integer_part, 10, buf);

	buf[offset++] = '.';

	// @FIXME
	while (precision--) {
		value *= 10.0;
		buf[offset++] = '0' + char(value);
		value = value - floor(value);
	}

	return String(buf, offset, allocator);
}
