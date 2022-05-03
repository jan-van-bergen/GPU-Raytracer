#pragma once
#include "Assertion.h"
#include "Array.h"
#include "StringView.h"

#include "Util/Util.h"
#include "Util/StringUtil.h"

template<typename T>
struct Formatter {

};

struct Format {
	struct Spec {
		const char * fmt_end;
		const char * restart;

		// See: https://en.cppreference.com/w/cpp/utility/format/formatter
		// And: https://docs.python.org/3/library/string.html#formatspec
		char fill      = ' ';
		char align     = 0;
		char sign      = '-';
		bool alternate = false;
		bool zero_pad  = false;
		int  width     = -1;
		int  precision = -1;
		char type      = 's';
	};

	Array<char> data;

	Format(Allocator * allocator = nullptr) : data(allocator) { }

	NON_COPYABLE(Format);
	NON_MOVEABLE(Format);

	~Format() { }

	String format(StringView fmt) {
		append(fmt);
		append('\0');
		return String(std::move(data));
	}

	template<typename Arg, typename ... Args>
	String format(StringView fmt, const Arg & arg, const Args & ... args) {
		Spec spec = parse_fmt(fmt);
		ASSERT(spec.fmt_end != fmt.end);
		append(StringView { fmt.start, spec.fmt_end });

		String str = Formatter<Arg>::format(spec, data.allocator, arg);

		if (spec.align == 0) {
			spec.align = '<'; // Default for non-integer/float
		}

		// Pre-alignment
		if (spec.width != -1 && str.size() < spec.width) {
			if (spec.align == '>') {
				append(spec.fill, spec.width - str.size());
			} else if (spec.align == '^') {
				append(spec.fill, (spec.width - str.size()) / 2);
			}
		}

		append(str.view());

		// Post-alignment
		if (spec.width != -1 && str.size() < spec.width) {
			if (spec.align == '<') {
				append(spec.fill, spec.width - str.size());
			} else if (spec.align == '^') {
				append(spec.fill, spec.width - str.size() - (spec.width - str.size()) / 2);
			}
		}

		// Recurse on the remaining fmt string using the next argument
		return format(StringView { spec.restart, fmt.end }, args...);
	}

private:
	void append(StringView str) {
		data.push_back(str.data(), str.size());
	}

	void append(char c, int n = 1) {
		for (int i = 0; i < n; i++) {
			data.push_back(c);
		}
	}

	Spec parse_fmt(StringView fmt) const;
};

template<size_t N, typename T>
struct Formatter<T[N]> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, const T * value) {
		return Formatter<const T *>::format(fmt_spec, allocator, value);
	}
};

template<>
struct Formatter<char> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, char value) {
		if (fmt_spec.type == 'd') {
			if (fmt_spec.align == 0) {
				fmt_spec.align = '>';
			}
			return Util::to_string(int64_t(value));
		} else {
			return String(&value, 1);
		}
	}
};

template<>
struct Formatter<StringView> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, StringView value) {
		return value;
	}
};

template<>
struct Formatter<String> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, const String & value) {
		return value;
	}
};

template<>
struct Formatter<const char *> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, const char * value) {
		return String(value, allocator);
	}
};

template<>
struct Formatter<bool> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, bool value) {
		if (fmt_spec.type == 'd') {
			if (fmt_spec.align == 0) {
				fmt_spec.align = '>';
			}
			return String(value ? "1"_sv : "0"_sv, allocator);
		} else {
			return Util::to_string(value, allocator);
		}
	}
};

template<>
struct Formatter<int64_t> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, int64_t value) {
		if (fmt_spec.align == 0) {
			fmt_spec.align = '>';
		}

		int64_t base = 10;
		if (fmt_spec.type == 'b') {
			base = 2;
		} else if (fmt_spec.type == 'x' || fmt_spec.type == 'X') {
			base = 16;
		}

		return Util::to_string(value, base, allocator);
	}
};

template<>
struct Formatter<int> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, int value) {
		return Formatter<int64_t>::format(fmt_spec, allocator, value);
	}
};

template<>
struct Formatter<uint64_t> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, uint64_t value) {
		if (fmt_spec.align == 0) {
			fmt_spec.align = '>';
		}

		uint64_t base = 10;
		if (fmt_spec.type == 'b') {
			base = 2;
		} else if (fmt_spec.type == 'x' || fmt_spec.type == 'X') {
			base = 16;
		}

		return Util::to_string(value, base, allocator);
	}
};

template<>
struct Formatter<unsigned> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, unsigned value) {
		return Formatter<int64_t>::format(fmt_spec, allocator, value);
	}
};

template<>
struct Formatter<double> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, double value) {
		if (fmt_spec.align == 0) {
			fmt_spec.align = '>';
		}
		return Util::to_string(value, allocator);
	}
};

template<>
struct Formatter<float> {
	static String format(Format::Spec & fmt_spec, Allocator * allocator, float value) {
		if (fmt_spec.align == 0) {
			fmt_spec.align = '>';
		}
		return Util::to_string(double(value), allocator);
	}
};
