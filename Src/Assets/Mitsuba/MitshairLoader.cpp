#include "MitshairLoader.h"

#include "Core/Array.h"

#include "Math/Quaternion.h"

#include "Renderer/Triangle.h"

Array<Triangle> MitshairLoader::load(const String & filename, SourceLocation location_in_mitsuba_file, float radius) {
	String file = IO::file_read(filename);

	Parser parser(file.view(), filename.view());

	Array<Vector3> hair_vertices;
	Array<int>     hair_strand_lengths;

	size_t triangle_count = 0;
	int strand_size = 0;

	if (parser.match("BINARY_HAIR")) { // Binary format
		unsigned num_vertices = parser.parse_binary<unsigned>();

		while (parser.cur < parser.end) {
			float x = parser.parse_binary<float>();
			if (isinf(x)) { // +INF marks beginning of new hair strand
				hair_strand_lengths.push_back(strand_size);
				triangle_count += strand_size - 1;
				strand_size = 0;
			} else {
				float y = parser.parse_binary<float>();
				float z = parser.parse_binary<float>();
				hair_vertices.emplace_back(x, y, z);
				strand_size++;
			}
		}
	} else { // ASCII format
		while (parser.cur < parser.end) {
			if (is_newline(*parser.cur)) { // Empty line marks beginning of new hair strand
				hair_strand_lengths.push_back(strand_size);
				triangle_count +=strand_size - 1;
				strand_size = 0;
			} else {
				float x = parser.parse_float(); parser.skip_whitespace();
				float y = parser.parse_float(); parser.skip_whitespace();
				float z = parser.parse_float(); parser.skip_whitespace();
				hair_vertices.emplace_back(x, y, z);
				strand_size++;
			}
			parser.parse_newline();
		}
	}

	if (strand_size > 0) {
		triangle_count += strand_size - 1;
	}

	triangle_count *= 2;
	Array<Triangle> triangles(triangle_count);

	int current_triangle = 0;

	struct Segment {
		Vector3 begin;
		Vector3 end;
	};
	Segment prev_segment = { };
	Segment curr_segment = { };

	size_t hair_index = 0;

	for (int s = 0; s < hair_strand_lengths.size(); s++) {
		int strand_size = hair_strand_lengths[s];

		const Vector3 * strand = &hair_vertices[hair_index];
		hair_index += strand_size;

		if (strand_size < 2) {
			WARNING(location_in_mitsuba_file, "WARNING: A hair strand was defined with less than 2 vertices!\n");
			triangle_count -= 2 * strand_size;
			continue;
		}

		float angle = PI * float(rand()) / float(RAND_MAX);

		Vector3 direction  = Vector3::normalize(strand[1] - strand[0]);
		Vector3 orthogonal = Quaternion::axis_angle(direction, angle) * Math::orthogonal(direction);

		prev_segment.begin = strand[0] + radius * orthogonal;
		prev_segment.end   = strand[0] - radius * orthogonal;

		for (int v = 1; v < strand_size; v++) {
			Vector3 direction = Vector3::normalize(strand[v] - strand[v-1]);
			Vector3 orthogonal;
			if (isnan(direction.x + direction.y + direction.z)) {
				orthogonal = Vector3(1.0f, 0.0f, 0.0f);
			} else {
				orthogonal = Quaternion::axis_angle(direction, angle) * Math::orthogonal(direction);
			}

			float r = Math::lerp(radius, 0.0f, float(v) / float(strand_size - 1));
			curr_segment.begin = strand[v] + r * orthogonal;
			curr_segment.end   = strand[v] - r * orthogonal;

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

	return triangles;
}
