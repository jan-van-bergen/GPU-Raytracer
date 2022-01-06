#include "PLYLoader.h"

#include "Util/Array.h"
#include "Util/Parser.h"
#include "Util/StringView.h"

enum struct Format {
	ASCII,
	BINARY_LITTLE_ENDIAN,
	BINARY_BIG_ENDIAN
};

struct Property {
	struct Type {
		enum struct Kind {
			INT8,
			INT16,
			INT32,
			UINT8,
			UINT16,
			UINT32,
			FLOAT32,
			FLOAT64,
			LIST
		} kind;
		union {
			struct {
				Kind size_type_kind;
				Kind list_type_kind;
			} list;
			struct {
				StringView name;
			} custom;
		};
	} type;

	enum struct Kind {
		X,
		Y,
		Z,
		NX,
		NY,
		NZ,
		U,
		V,
		IGNORED, // Unknown properties are ignored
		VERTEX_INDEX
	} kind;
};

struct Element {
	struct Type {
		enum struct Kind {
			VERTEX,
			FACE
		} kind;

		StringView custom_name;
	} type;

	int count;

	static constexpr int MAX_PROPERTIES = 16;

	Property properties[MAX_PROPERTIES];
	int      property_count;
};

static Property::Type parse_property_type(Parser & parser) {
	parser.skip_whitespace();

	Property::Type type = { };

	if (parser.match("int8") || parser.match("char")) {
		type.kind = Property::Type::Kind::INT8;
	} else if (parser.match("int16") || parser.match("short")) {
		type.kind = Property::Type::Kind::INT16;
	} else if (parser.match("int32") || parser.match("int")) {
		type.kind = Property::Type::Kind::INT32;
	} else if (parser.match("uint8") || parser.match("uchar")) {
		type.kind = Property::Type::Kind::UINT8;
	} else if (parser.match("uint16") || parser.match("ushort")) {
		type.kind = Property::Type::Kind::UINT16;
	} else if (parser.match("uint32") || parser.match("uint")) {
		type.kind = Property::Type::Kind::UINT32;
	} else if (parser.match("float32") || parser.match("float")) {
		type.kind = Property::Type::Kind::FLOAT32;
	} else if (parser.match("float64") || parser.match("double")) {
		type.kind = Property::Type::Kind::FLOAT64;
	} else if (parser.match("list")) {
		type.kind = Property::Type::Kind::LIST;
		type.list.size_type_kind = parse_property_type(parser).kind;
		type.list.list_type_kind = parse_property_type(parser).kind;
	} else {
		ERROR(parser.location, "Invalid type!\n");
	}

	return type;
}

template<typename T>
static T parse_value(Parser & parser, Format format) {
	const char * start = parser.cur;
	parser.cur += sizeof(T);

	T value = { };

	// NOTE: Assumes machine is little endian!
	switch (format) {
		case Format::BINARY_LITTLE_ENDIAN: memcpy(&value, start, sizeof(T)); break;
		case Format::BINARY_BIG_ENDIAN: {
			char       * dst = reinterpret_cast<char *>(&value);
			const char * src = start;
			for (int i = 0; i < sizeof(T); i++) {
				dst[i] = src[sizeof(T) - 1 - i];
			}
			break;
		}
		default: abort();
	}

	return value;
}

template<typename T>
static T parse_property_value(Parser & parser, Property::Type::Kind kind, Format format) {
	if (format == Format::ASCII) {
		parser.skip_whitespace();
		if (kind == Property::Type::Kind::FLOAT32 || kind == Property::Type::Kind::FLOAT64) {
			return parser.parse_float();
		} else {
			return parser.parse_int();
		}
	} else {
		// Parse as binary data
		switch (kind) {
			case Property::Type::Kind::INT8:    return parse_value<int8_t>  (parser, format); break;
			case Property::Type::Kind::INT16:	return parse_value<int16_t> (parser, format); break;
			case Property::Type::Kind::INT32:	return parse_value<int32_t> (parser, format); break;
			case Property::Type::Kind::UINT8:	return parse_value<uint8_t> (parser, format); break;
			case Property::Type::Kind::UINT16:	return parse_value<uint16_t>(parser, format); break;
			case Property::Type::Kind::UINT32:	return parse_value<uint32_t>(parser, format); break;
			case Property::Type::Kind::FLOAT32:	return parse_value<float>   (parser, format); break;
			case Property::Type::Kind::FLOAT64:	return parse_value<double>  (parser, format); break;

			default: ERROR(parser.location, "Invalid property type!\n"); break;
		}
	}
}

void PLYLoader::load(const String & filename, Triangle *& triangles, int & triangle_count) {
	String file = Util::file_read(filename);

	Parser parser;
	parser.init(file.view(), filename.view());

	parser.expect("ply");
	parser.skip_whitespace();
	parser.parse_newline();
	parser.expect("format");
	parser.skip_whitespace();

	Format format;

	if (parser.match("ascii")) {
		format = Format::ASCII;
	} else if (parser.match("binary_little_endian")) {
		format = Format::BINARY_LITTLE_ENDIAN;
	} else if (parser.match("binary_big_endian")) {
		format = Format::BINARY_BIG_ENDIAN;
	} else {
		ERROR(parser.location, "Invalid PLY format!\n");
	}
	parser.skip_whitespace();

	int version_major = parser.parse_int();
	parser.expect('.');
	int version_minor = parser.parse_int();
	parser.skip_whitespace_or_newline();

	if (version_major != 1 || version_minor != 0) {
		WARNING(parser.location, "PLY format version is not 1.0!\n");
	}

	Array<Element> elements;

	while (!parser.match("end_header")) {
		if (parser.match("comment")) {
			while (!is_newline(*parser.cur)) parser.advance();
		} else if (parser.match("element")) {
			parser.skip_whitespace();

			Element element = { };

			if (parser.match("vertex ")) {
				element.type.kind = Element::Type::Kind::VERTEX;
			} else if (parser.match("face ")) {
				element.type.kind = Element::Type::Kind::FACE;
			} else {
				StringView element_name = parser.parse_identifier();
				ERROR(parser.location, "Unsupported element type '%.*s'!\n", FMT_STRINGVIEW(element_name));
			}
			parser.skip_whitespace();

			element.count = parser.parse_int();
			elements.push_back(element);
		} else if (parser.match("property")) {
			parser.skip_whitespace();

			if (elements.size() == 0) {
				ERROR(parser.location, "Property defined without element!\n");
			}
			Element & element = elements.back();

			if (element.property_count == Element::MAX_PROPERTIES) {
				ERROR(parser.location, "Maximum number of properties (%i) exceeded!\n", Element::MAX_PROPERTIES);
			}
			Property & property = element.properties[element.property_count++];

			property.type = parse_property_type(parser);

			StringView name = parser.parse_identifier();
			if (name == "x") {
				property.kind = Property::Kind::X;
			} else if (name == "y") {
				property.kind = Property::Kind::Y;
			} else if (name == "z") {
				property.kind = Property::Kind::Z;
			} else if (name == "nx") {
				property.kind = Property::Kind::NX;
			} else if (name == "ny") {
				property.kind = Property::Kind::NY;
			} else if (name == "nz") {
				property.kind = Property::Kind::NZ;
			} else if (name == "u" || name == "s") {
				property.kind = Property::Kind::U;
			} else if (name == "v" || name == "t") {
				property.kind = Property::Kind::V;
			} else if (name == "vertex_index" || name == "vertex_indices") {
				property.kind = Property::Kind::VERTEX_INDEX;
			} else {
				property.kind = Property::Kind::IGNORED;
				WARNING(parser.location, "Unknown property '%.*s'!\n", FMT_STRINGVIEW(name));
			}
		}

		parser.skip_whitespace();
		parser.parse_newline();
	}

	parser.skip_whitespace();
	parser.parse_newline();

	Array<Vector3> positions;
	Array<Vector2> tex_coords;
	Array<Vector3> normals;

	Array<Triangle> tris;

	for (int e = 0; e < elements.size(); e++) {
		const Element & element = elements[e];

		switch (element.type.kind) {
			case Element::Type::Kind::VERTEX: {
				for (int i = 0; i < element.count; i++) {
					float vertex[9] = { }; // x, y, z, nx, ny, nz, u, v, ignored

					for (int p = 0; p < element.property_count; p++) {
						const Property & property = element.properties[p];

						vertex[int(property.kind)] = parse_property_value<float>(parser, property.type.kind, format);
					}

					positions .emplace_back(vertex[0], vertex[1], vertex[2]);
					normals   .emplace_back(vertex[3], vertex[4], vertex[5]);
					tex_coords.emplace_back(vertex[6], 1.0f - vertex[7]);

					if (format == Format::ASCII) {
						parser.skip_whitespace();
						parser.parse_newline();
					}
				}

				break;
			}

			case Element::Type::Kind::FACE: {
				tris.reserve(element.count); // Expect as many triangles as there are faces, but there could be more (if faces have more than 3 vertices)
				for (int i = 0; i < element.count; i++) {
					for (int p = 0; p < element.property_count; p++) {
						const Property & property = element.properties[p];

						if (property.kind != Property::Kind::VERTEX_INDEX || property.type.kind != Property::Type::Kind::LIST) {
							break;
						}

						size_t size = parse_property_value<size_t>(parser, property.type.list.size_type_kind, format);

						if (size <= 2) {
							ERROR(parser.location, "A Triangle needs at least 3 indices!\n");
						}

						size_t elem_0 = parse_property_value<size_t>(parser, property.type.list.list_type_kind, format);
						size_t elem_1 = parse_property_value<size_t>(parser, property.type.list.list_type_kind, format);

						Vector3 pos[3] = { positions [elem_0], positions [elem_1] };
						Vector2 tex[3] = { tex_coords[elem_0], tex_coords[elem_1] };
						Vector3 nor[3] = { normals   [elem_0], normals   [elem_1] };

						for (size_t i = 2; i < size; i++) {
							size_t elem_2 = parse_property_value<size_t>(parser, property.type.list.list_type_kind, format);
							pos[2] = positions [elem_2];
							tex[2] = tex_coords[elem_2];
							nor[2] = normals   [elem_2];

							Triangle triangle = { };
							triangle.position_0  = pos[0];
							triangle.position_1  = pos[1];
							triangle.position_2  = pos[2];
							triangle.tex_coord_0 = tex[0];
							triangle.tex_coord_1 = tex[1];
							triangle.tex_coord_2 = tex[2];
							triangle.normal_0    = nor[0];
							triangle.normal_1    = nor[1];
							triangle.normal_2    = nor[2];
							triangle.init();

							tris.push_back(triangle);

							pos[1] = pos[2];
							tex[1] = tex[2];
							nor[1] = nor[2];
						}
					}

					if (format == Format::ASCII && !parser.reached_end()) {
						parser.skip_whitespace();
						parser.parse_newline();
					}
				}

				break;
			}

			default: abort();
		}
	}

	assert(parser.reached_end());

	triangle_count = tris.size();
	triangles      = new Triangle[triangle_count];
	memcpy(triangles, tris.data(), triangle_count * sizeof(Triangle));
}
