#include "StringUtil.h"

StringView Util::get_directory(StringView filename) {
	const char * last_slash = find_last_after(filename, StringView::from_c_str("/\\"));

	if (last_slash != filename.end) {
		return StringView { filename.start, last_slash };
	} else {
		return StringView::from_c_str("./");
	}
}

StringView Util::remove_directory(StringView filename) {
	const char * last_slash = find_last_after(filename, StringView::from_c_str("/\\"));

	if (last_slash != filename.end) {
		return StringView { last_slash, filename.end };
	} else {
		return filename;
	}
}

StringView Util::get_file_extension(StringView filename) {
	return StringView { find_last_after(filename, StringView::from_c_str(".")), filename.end };
}

StringView Util::substr(StringView str, size_t offset, size_t len) {
	if (offset + len >= str.length()) {
		len = str.length() - offset;
	}
	return { str.start + offset, str.start + offset + len };
}

String Util::combine_stringviews(StringView a, StringView b) {
	String filename_abs = String(a.length() + b.length());

	memcpy(filename_abs.data(),              a.start, a.length());
	memcpy(filename_abs.data() + a.length(), b.start, b.length());
	filename_abs[a.length() + b.length()] = '\0';

	return filename_abs;
}

const char * Util::find_last_after(StringView haystack, StringView needles) {
	const char * cur        = haystack.start;
	const char * last_match = nullptr;

	while (cur < haystack.end) {
		for (int i = 0; i < needles.length(); i++) {
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

	const char * cur = haystack.start;
	while (cur < haystack.end - needle.length()) {
		bool match = true;
		for (int i = 0; i < needle.length(); i++) {
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
