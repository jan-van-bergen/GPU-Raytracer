#include "SerializedLoader.h"

#include <miniz/miniz.h>

#include "Renderer/Triangle.h"

#include "XMLParser.h"

Array<Triangle> SerializedLoader::load(const String & filename, Allocator * allocator, SourceLocation location_in_mitsuba_file, int shape_index) {
	String serialized = IO::file_read(filename, allocator);
	Parser serialized_parser(serialized.view(), filename.view());

	uint16_t file_format_id = serialized_parser.parse_binary<uint16_t>();
	if (file_format_id != 0x041c) {
		ERROR(location_in_mitsuba_file, "ERROR: Serialized file '{}' does not start with format ID 0x041c!\n", filename);
	}

	uint16_t file_version = serialized_parser.parse_binary<uint16_t>();

	// Read the End-of-File Dictionary
	serialized_parser.seek(serialized.size() - sizeof(uint32_t));
	uint32_t num_meshes = serialized_parser.parse_binary<uint32_t>();
	uint64_t eof_dictionary_offset = 0;;

	Array<uint64_t> mesh_offsets(num_meshes + 1, allocator);

	if (file_version <= 3) {
		// Version 0.3.0 and earlier use 32 bit mesh offsets
		eof_dictionary_offset = serialized.size() - sizeof(uint32_t) - num_meshes * sizeof(uint32_t);
		serialized_parser.seek(eof_dictionary_offset);

		for (uint32_t i = 0; i < num_meshes; i++) {
			mesh_offsets[i] = serialized_parser.parse_binary<uint32_t>();
		}
	} else {
		// Version 0.4.0 and later use 64 bit mesh offsets
		eof_dictionary_offset = serialized.size() - sizeof(uint32_t) - num_meshes * sizeof(uint64_t);
		serialized_parser.seek(eof_dictionary_offset);

		for (uint32_t i = 0; i < num_meshes; i++) {
			mesh_offsets[i] = serialized_parser.parse_binary<uint64_t>();
		}
	}

	mesh_offsets[num_meshes] = eof_dictionary_offset;
	ASSERT(mesh_offsets[0] == 0);

	// Decompress stream for this Mesh
	mz_ulong num_bytes = mesh_offsets[shape_index + 1] - mesh_offsets[shape_index] - 4;

	mz_ulong deserialized_length = 3 * num_bytes;
	String   deserialized = { };

	while (true) {
		deserialized = String(deserialized_length);

		int status = uncompress(
			reinterpret_cast<      unsigned char *>(deserialized.data()), &deserialized_length,
			reinterpret_cast<const unsigned char *>(serialized.data() + mesh_offsets[shape_index] + 4), num_bytes
		);

		if (status == MZ_BUF_ERROR) {
			deserialized_length *= 2;
		} else if (status == MZ_OK) {
			break;
		} else {
			ERROR(location_in_mitsuba_file, "ERROR: Failed to decompress serialized mesh #{} in file '{}'!\n{}\n", shape_index, filename, mz_error(status));
		}
	}

	Parser deserialized_parser(deserialized.view(), filename.view());

	// Read flags field
	uint32_t flags = deserialized_parser.parse_binary<uint32_t>();
	bool flag_has_normals      = flags & 0x0001;
	bool flag_has_tex_coords   = flags & 0x0002;
	bool flag_has_colours      = flags & 0x0008;
	bool flag_use_face_normals = flags & 0x0010;
	bool flag_single_precision = flags & 0x1000;
	bool flag_double_precision = flags & 0x2000;

	if (file_version <= 3) {
		flag_single_precision = true;
	} else {
		// Read null terminated name
		deserialized_parser.parse_c_str();
	}

	// Read number of vertices and triangles1
	uint64_t num_vertices  = deserialized_parser.parse_binary<uint64_t>();
	uint64_t num_triangles = deserialized_parser.parse_binary<uint64_t>();

	if (num_vertices == 0 || num_triangles == 0) {
		WARNING(location_in_mitsuba_file, "WARNING: Serialized Mesh defined without vertices or triangles!\n");
		return { };
	}

	// Check if num_vertices fits inside a uint32_t to determine whether indices use 32 or 64 bits
	bool fits_in_32_bits = num_vertices <= 0xffffffff;

	size_t element_size;
	if (flag_single_precision) {
		element_size = sizeof(float);
	} else if (flag_double_precision) {
		element_size = sizeof(double);
	} else {
		ERROR(location_in_mitsuba_file, "ERROR: Neither single nor double precision specified!\n");
	}

	const char * vertex_positions = deserialized_parser.cur;
	deserialized_parser.advance(num_vertices * 3 * element_size);

	const char * vertex_normals = deserialized_parser.cur;
	if (flag_has_normals) {
		deserialized_parser.advance(num_vertices * 3 * element_size);
	}

	const char * vertex_tex_coords = deserialized_parser.cur;
	if (flag_has_tex_coords) {
		deserialized_parser.advance(num_vertices * 2 * element_size);
	}

	// Vertex colours, not used
	if (flag_has_colours) {
		deserialized_parser.advance(num_vertices * 3 * element_size);
	}

	const char * indices = deserialized_parser.cur;

	// Reads a Vector3 from a buffer with the appropriate precision
	auto read_vector3 = [flag_single_precision](const char * buffer, uint64_t index) {
		Vector3 result;
		if (flag_single_precision) {
			result.x = reinterpret_cast<const float *>(buffer)[3*index];
			result.y = reinterpret_cast<const float *>(buffer)[3*index + 1];
			result.z = reinterpret_cast<const float *>(buffer)[3*index + 2];
		} else {
			result.x = float(reinterpret_cast<const double *>(buffer)[3*index]);
			result.y = float(reinterpret_cast<const double *>(buffer)[3*index + 1]);
			result.z = float(reinterpret_cast<const double *>(buffer)[3*index + 2]);
		}
		return result;
	};

	// Reads a Vector2 from a buffer with the appropriate precision
	auto read_vector2 = [flag_single_precision](const char * buffer, uint64_t index) {
		Vector2 result;
		if (flag_single_precision) {
			result.x = reinterpret_cast<const float *>(buffer)[2*index];
			result.y = reinterpret_cast<const float *>(buffer)[2*index + 1];
		} else {
			result.x = float(reinterpret_cast<const double *>(buffer)[2*index]);
			result.y = float(reinterpret_cast<const double *>(buffer)[2*index + 1]);
		}
		return result;
	};

	// Reads a 32 or 64 bit indx from the specified buffer
	auto read_index = [fits_in_32_bits](const char * buffer, uint64_t index) -> uint64_t {
		if (fits_in_32_bits) {
			return reinterpret_cast<const uint32_t *>(buffer)[index];
		} else {
			return reinterpret_cast<const uint64_t *>(buffer)[index];
		}
	};

	// Construct triangles
	Array<Triangle> triangles(num_triangles);

	for (size_t t = 0; t < num_triangles; t++) {
		uint64_t index_0 = read_index(indices, 3*t);
		uint64_t index_1 = read_index(indices, 3*t + 1);
		uint64_t index_2 = read_index(indices, 3*t + 2);

		triangles[t].position_0 = read_vector3(vertex_positions, index_0);
		triangles[t].position_1 = read_vector3(vertex_positions, index_1);
		triangles[t].position_2 = read_vector3(vertex_positions, index_2);

		if (flag_use_face_normals) {
			Vector3 geometric_normal = Vector3::normalize(Vector3::cross(
				triangles[t].position_1 - triangles[t].position_0,
				triangles[t].position_2 - triangles[t].position_0
			));
			triangles[t].normal_0 = geometric_normal;
			triangles[t].normal_1 = geometric_normal;
			triangles[t].normal_2 = geometric_normal;
		} else if (flag_has_normals) {
			triangles[t].normal_0 = read_vector3(vertex_normals, index_0);
			triangles[t].normal_1 = read_vector3(vertex_normals, index_1);
			triangles[t].normal_2 = read_vector3(vertex_normals, index_2);
		}

		if (flag_has_tex_coords) {
			triangles[t].tex_coord_0 = read_vector2(vertex_tex_coords, index_0);
			triangles[t].tex_coord_1 = read_vector2(vertex_tex_coords, index_1);
			triangles[t].tex_coord_2 = read_vector2(vertex_tex_coords, index_2);
		}

		triangles[t].init();
	}

	return triangles;
}
