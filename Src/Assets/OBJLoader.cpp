#include "OBJLoader.h"

#include "Math/Vector2.h"
#include "Math/Vector3.h"

#include "Util/Array.h"
#include "Util/Parser.h"
#include "Util/String.h"

static float parse_float(Parser & parser) {
	parser.skip_whitespace();
	return parser.parse_float();
}

static int parse_int(Parser & parser) {
	parser.skip_whitespace();
	return parser.parse_int();
}

static Vector2 parse_vector2(Parser & parser) {
	float x = parse_float(parser);
	float y = parse_float(parser);
	return Vector2(x, y);
}

static Vector3 parse_vector3(Parser & parser) {
	float x = parse_float(parser);
	float y = parse_float(parser);
	float z = parse_float(parser);
	return Vector3(x, y, z);
}

struct Index {
	int v, t, n; // Position, texcoord, normal indices
};

struct Face {
	Index indices[3]; // Always a triangular face
};

static Index parse_index(Parser & parser) {
	Index index = { };

	index.v = parse_int(parser);

	if (parser.match('/')) {
		if (!parser.match('/')) {
			index.t = parse_int(parser);
			parser.expect('/');
		}
		index.n = parse_int(parser);
	}

	return index;
}

static void parse_face(Parser & parser, Array<Face> & faces) {
	// Parse first triangular face
	Index index_0 = parse_index(parser);
	Index index_1 = parse_index(parser);
	Index index_2 = parse_index(parser);
	faces.emplace_back(index_0, index_1, index_2);

	// Triangulate any further vertices in the face
	Index prev_index = index_2;

	while (true) {
		parser.skip_whitespace();

		if (parser.reached_end() || !(*parser.cur == '-' || is_digit(*parser.cur))) break;

		Index curr_index = parse_index(parser);
		faces.emplace_back(index_0, prev_index, curr_index);

		prev_index = curr_index;
	}
}

struct OBJFile {
	Array<Vector3> positions;
	Array<Vector2> tex_coords;
	Array<Vector3> normals;

	Array<Face> faces;
};

static OBJFile parse_obj(const char * filename) {
	int          file_length;
	const char * file = Util::file_read(filename, file_length);

	OBJFile obj = { };

	Parser parser = { };
	parser.init(file, file + file_length, filename);

	while (!parser.reached_end()) {
		if (parser.match('#') || parser.match("o ")) {
			while (!parser.reached_end() && !is_newline(*parser.cur)) {
				parser.advance();
			}
		}
		else if (parser.match("v "))  obj.positions .push_back(parse_vector3(parser));
		else if (parser.match("vt ")) obj.tex_coords.push_back(parse_vector2(parser));
		else if (parser.match("vn ")) obj.normals   .push_back(parse_vector3(parser));
		else if (parser.match("f "))  parse_face(parser, obj.faces);
		else {
			while (!parser.reached_end() && !is_newline(*parser.cur)) {
				parser.advance();
			}
		}

		parser.skip_whitespace();
		parser.match('\r');
		parser.expect('\n');
	}

	delete [] file;

	return obj;
}

bool OBJLoader::load(const char * filename, Triangle *& triangles, int & triangle_count) {
	OBJFile obj = parse_obj(filename);

	triangle_count = obj.faces.size();
	triangles      = new Triangle[triangle_count];

	for (int f = 0; f < obj.faces.size(); f++) {
		const Face & face = obj.faces[f];

		Vector3 positions [3] = { };
		Vector2 tex_coords[3] = { };
		Vector3 normals   [3] = { };

		for (int i = 0; i < 3; i++) {
			int v = face.indices[i].v;
			int t = face.indices[i].t;
			int n = face.indices[i].n;

			auto get_index = [](int array_size, int index) {
				int result = INVALID;
				if (array_size != 0) {
					if (index > 0) {
						result = index - 1;
					} else if (index < 0) {
						result = array_size + index; // Negative indices are relative to end
					}

					// Check if the index is valid given the Array size
					if (result < 0 || result >= array_size) {
						result = INVALID;
					}
				}
				return result;
			};

			int index_v = get_index(obj.positions .size(), v);
			int index_t = get_index(obj.tex_coords.size(), t);
			int index_n = get_index(obj.normals   .size(), n);

			if (index_v != INVALID) {
				positions[i] = obj.positions[index_v];
			}
			if (index_t != INVALID) {
				tex_coords[i] = obj.tex_coords[index_t];
				tex_coords[i].y = 1.0f - tex_coords[i].y; // Flip uv along v
			}
			if (index_n != INVALID) {
				normals[i] = obj.normals[index_n];
			}
		}

		triangles[f].position_0 = positions[0];
		triangles[f].position_1 = positions[1];
		triangles[f].position_2 = positions[2];
		triangles[f].normal_0 = normals[0];
		triangles[f].normal_1 = normals[1];
		triangles[f].normal_2 = normals[2];
		triangles[f].tex_coord_0 = tex_coords[0];
		triangles[f].tex_coord_1 = tex_coords[1];
		triangles[f].tex_coord_2 = tex_coords[2];
		triangles[f].init();
	}

	printf("Loaded OBJ %s from disk (%i triangles)\n", filename, triangle_count);

	return true;
}
