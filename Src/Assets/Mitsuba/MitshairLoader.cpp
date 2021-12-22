#include "MitshairLoader.h"

#include "Math/Quaternion.h"

#include "Pathtracer/Triangle.h"

#include "Util/Array.h"

void MitshairLoader::load(const char * filename, SourceLocation location_in_mitsuba_file, Triangle *& triangles, int & triangle_count, float radius) {
	int          file_length;
	const char * file = Util::file_read(filename, file_length);

	Parser parser = { };
	parser.init(file, file + file_length, filename);

	Array<Array<Vector3>> hairs;
	Array<Vector3>        strand;

	triangle_count = 0;

	if (parser.match("BINARY_HAIR")) { // Binary format
		unsigned num_vertices = parser.parse_binary<unsigned>();

		while (parser.cur < parser.end) {
			float x = parser.parse_binary<float>();
			if (isinf(x)) { // +INF marks beginning of new hair strand
				triangle_count += strand.size() - 1;
				hairs.push_back(strand);
				strand.clear();
			} else {
				float y = parser.parse_binary<float>();
				float z = parser.parse_binary<float>();
				strand.emplace_back(x, y, z);
			}
		}
	} else { // ASCII format
		while (parser.cur < parser.end) {
			if (is_newline(*parser.cur)) { // Empty line marks beginning of new hair strand
				triangle_count += strand.size() - 1;
				hairs.push_back(strand);
				strand.clear();
			} else {
				float x = parser.parse_float(); parser.skip_whitespace();
				float y = parser.parse_float(); parser.skip_whitespace();
				float z = parser.parse_float(); parser.skip_whitespace();
				strand.emplace_back(x, y, z);
			}
			parser.parse_newline();
		}
	}

	if (strand.size() > 0) {
		triangle_count += strand.size() - 1;
		hairs.push_back(strand);
	}

	delete [] file;

	triangle_count *= 2;
	triangles       = new Triangle[triangle_count];

	int current_triangle = 0;

	struct Segment {
		Vector3 begin;
		Vector3 end;
	};
	Segment prev_segment = { };
	Segment curr_segment = { };

	for (int h = 0; h < hairs.size(); h++) {
		const Array<Vector3> & strand = hairs[h];

		if (strand.size() < 2) {
			WARNING(location_in_mitsuba_file, "A hair strand requires at least 2 vertices!\n");
			triangle_count -= 2 * strand.size();
			continue;
		}

		float angle = PI * float(rand()) / float(RAND_MAX);

		Vector3 direction  = Vector3::normalize(strand[1] - strand[0]);
		Vector3 orthogonal = Quaternion::axis_angle(direction, angle) * Math::orthogonal(direction);

		prev_segment.begin = strand[0] + radius * orthogonal;
		prev_segment.end   = strand[0] - radius * orthogonal;

		for (int s = 1; s < strand.size(); s++) {
			Vector3 direction = Vector3::normalize(strand[s] - strand[s-1]);
			Vector3 orthogonal;
			if (isnan(direction.x + direction.y + direction.z)) {
				orthogonal = Vector3(1.0f, 0.0f, 0.0f);
			} else {
				orthogonal = Quaternion::axis_angle(direction, angle) * Math::orthogonal(direction);
			}

			float r = Math::lerp(radius, 0.0f, float(s) / float(strand.size() - 1));
			curr_segment.begin = strand[s] + r * orthogonal;
			curr_segment.end   = strand[s] - r * orthogonal;

			triangles[current_triangle].position_0  = prev_segment.begin;
			triangles[current_triangle].position_1  = prev_segment.end;
			triangles[current_triangle].position_2  = curr_segment.begin;
			triangles[current_triangle].normal_0    = Vector3(0.0f);
			triangles[current_triangle].normal_1    = Vector3(0.0f);
			triangles[current_triangle].normal_2    = Vector3(0.0f);
			triangles[current_triangle].tex_coord_0 = Vector2(0.0f, 0.0f);
			triangles[current_triangle].tex_coord_1 = Vector2(1.0f, 0.0f);
			triangles[current_triangle].tex_coord_2 = Vector2(0.0f, 1.0f);
			triangles[current_triangle].init();
			current_triangle++;

			triangles[current_triangle].position_0  = prev_segment.end;
			triangles[current_triangle].position_1  = curr_segment.end;
			triangles[current_triangle].position_2  = curr_segment.begin;
			triangles[current_triangle].normal_0    = Vector3(0.0f);
			triangles[current_triangle].normal_1    = Vector3(0.0f);
			triangles[current_triangle].normal_2    = Vector3(0.0f);
			triangles[current_triangle].tex_coord_0 = Vector2(0.0f, 0.0f);
			triangles[current_triangle].tex_coord_1 = Vector2(1.0f, 0.0f);
			triangles[current_triangle].tex_coord_2 = Vector2(0.0f, 1.0f);
			triangles[current_triangle].init();
			current_triangle++;

			prev_segment = curr_segment;
		}
	}
}
