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
	Index index = { INVALID, INVALID, INVALID };

	index.v = parse_int(parser) - 1;

	if (parser.match('/')) {
		if (parser.match('/')) {
			index.n = parse_int(parser) - 1;
		} else {
			index.t = parse_int(parser) - 1;
			if (parser.match('/')) {
				index.n = parse_int(parser) - 1;
			}
		}
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

		if (parser.reached_end() || !is_digit(*parser.cur)) break;

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

	SourceLocation location = { };
	location.file = filename;
	location.line = 1;
	location.col  = 0;

	Parser parser = { };
	parser.init(file, file + file_length, location);

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

			if (v != INVALID) {
				positions[i] = obj.positions[v];
			}
			if (t != INVALID) {
				tex_coords[i] = obj.tex_coords[t];
				tex_coords[i].y = 1.0f - tex_coords[i].y; // Flip uv along v
			}
			if (n != INVALID) {
				normals[i] = obj.normals[n];
			}
		}

		triangles[f].position_0 = positions[0];
		triangles[f].position_1 = positions[1];
		triangles[f].position_2 = positions[2];

		bool normal_0_invalid = Math::approx_equal(Vector3::length(normals[0]), 0.0f);
		bool normal_1_invalid = Math::approx_equal(Vector3::length(normals[1]), 0.0f);
		bool normal_2_invalid = Math::approx_equal(Vector3::length(normals[2]), 0.0f);

		// Replace zero normals with the geometric normal of defined by the Triangle
		if (normal_0_invalid || normal_1_invalid || normal_2_invalid) {
			Vector3 geometric_normal = Vector3::normalize(Vector3::cross(
				triangles[f].position_1 - triangles[f].position_0,
				triangles[f].position_2 - triangles[f].position_0
			));

			if (normal_0_invalid) normals[0] = geometric_normal;
			if (normal_1_invalid) normals[1] = geometric_normal;
			if (normal_2_invalid) normals[2] = geometric_normal;
		}

		triangles[f].normal_0 = normals[0];
		triangles[f].normal_1 = normals[1];
		triangles[f].normal_2 = normals[2];

		triangles[f].tex_coord_0 = tex_coords[0];
		triangles[f].tex_coord_1 = tex_coords[1];
		triangles[f].tex_coord_2 = tex_coords[2];

		triangles[f].calc_aabb();
	}

	printf("Loaded OBJ %s from disk (%i triangles)\n", filename, triangle_count);

	return true;
}
