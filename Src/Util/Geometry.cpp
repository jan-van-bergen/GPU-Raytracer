#include "Geometry.h"

void Geometry::rectangle(Triangle *& triangles, int & triangle_count, const Matrix4 & transform) {
	Vector3 vertex_0 = Matrix4::transform_position(transform, Vector3(-1.0f, -1.0f, 0.0f));
	Vector3 vertex_1 = Matrix4::transform_position(transform, Vector3(+1.0f, -1.0f, 0.0f));
	Vector3 vertex_2 = Matrix4::transform_position(transform, Vector3(+1.0f, +1.0f, 0.0f));
	Vector3 vertex_3 = Matrix4::transform_position(transform, Vector3(-1.0f, +1.0f, 0.0f));

	Vector3 normal = Matrix4::transform_direction(transform, Vector3(0.0f, 0.0f, 1.0f));

	Vector2 tex_coord_0 = Vector2(0.0f, 0.0f);
	Vector2 tex_coord_1 = Vector2(1.0f, 0.0f);
	Vector2 tex_coord_2 = Vector2(1.0f, 1.0f);
	Vector2 tex_coord_3 = Vector2(0.0f, 1.0f);

	triangle_count = 2;
	triangles = new Triangle[triangle_count];
		
	triangles[0].position_0 = vertex_0;
	triangles[0].position_1 = vertex_1;
	triangles[0].position_2 = vertex_2;
	triangles[0].normal_0 = normal;
	triangles[0].normal_1 = normal;
	triangles[0].normal_2 = normal;
	triangles[0].tex_coord_0 = tex_coord_0;
	triangles[0].tex_coord_1 = tex_coord_1;
	triangles[0].tex_coord_2 = tex_coord_2;
			
	triangles[1].position_0 = vertex_0;
	triangles[1].position_1 = vertex_2;
	triangles[1].position_2 = vertex_3;
	triangles[1].normal_0 = normal;
	triangles[1].normal_1 = normal;
	triangles[1].normal_2 = normal;
	triangles[1].tex_coord_0 = tex_coord_0;
	triangles[1].tex_coord_1 = tex_coord_2;
	triangles[1].tex_coord_2 = tex_coord_3;

	Vector3 vertices_0[3] = { vertex_0, vertex_1, vertex_2 };
	Vector3 vertices_1[3] = { vertex_0, vertex_2, vertex_3 };

	triangles[0].aabb = AABB::from_points(vertices_0, 3);
	triangles[1].aabb = AABB::from_points(vertices_1, 3);
}

void Geometry::disk(Triangle *& triangles, int & triangle_count, const Matrix4 & transform, int num_segments) {
	triangle_count = num_segments;
	triangles = new Triangle[triangle_count];

	Vector3 vertex_center = Matrix4::transform_position(transform, Vector3(0.0f, 0.0f, 0.0f));
	Vector3 vertex_prev   = Matrix4::transform_position(transform, Vector3(1.0f, 0.0f, 0.0f));
	
	Vector3 normal = Matrix4::transform_direction(transform, Vector3(0.0f, 0.0f, 1.0f));

	Vector2 uv_prev = Vector2(1.0f, 0.5f);

	float angle = TWO_PI / float(num_segments);
	float theta = 0.0f;

	for (int i = 0; i < num_segments; i++) {
		theta += angle;

		float cos_theta = cosf(theta);
		float sin_theta = sinf(theta);

		Vector3 vertex_curr = Matrix4::transform_position(transform, Vector3(cos_theta, sin_theta, 0.0f));

		Vector2 uv_curr = Vector2(0.5f + 0.5f * cos_theta, 0.5f + 0.5f * sin_theta);

		triangles[i].position_0 = vertex_prev;
		triangles[i].position_1 = vertex_curr;
		triangles[i].position_2 = vertex_center;

		triangles[i].normal_0 = normal;
		triangles[i].normal_1 = normal;
		triangles[i].normal_2 = normal;

		triangles[i].tex_coord_0 = uv_prev;
		triangles[i].tex_coord_0 = uv_curr;
		triangles[i].tex_coord_0 = Vector2(0.5f, 0.5f);
		
		Vector3 vertices[3] = { vertex_prev, vertex_curr, vertex_center };
		triangles[i].aabb = AABB::from_points(vertices, 3);
		
		vertex_prev = vertex_curr;
		uv_prev = uv_curr;
	}
}

void Geometry::sphere(Triangle *& triangles, int & triangle_count, const Matrix4 & transform, int num_subdivisions) {
	constexpr float x = 0.525731112119133606f;
	constexpr float z = 0.850650808352039932f; 

	static Vector3 icosahedron_vertices[12] = {    
		Vector3(-x, 0.0f, z), Vector3(x, 0.0f, z),  Vector3(-x, 0.0f, -z), Vector3(x, 0.0f, -z),    
		Vector3(0.0f, z, x),  Vector3(0.0f, z, -x), Vector3(0.0f, -z, x),  Vector3(0.0, -z, -x),    
		Vector3(z, x, 0.0f),  Vector3(-z, x, 0.0f), Vector3(z, -x, 0.0f),  Vector3(-z, -x, 0.0f) 
	};
	static int icosahedron_indices[20][3] = {
		{ 0, 4, 1 },  { 0, 9, 4 },  { 9, 5, 4 },  { 4, 5, 8 },  { 4, 8, 1 },    
		{ 8, 10, 1 }, { 8, 3, 10 }, { 5, 3, 8 },  { 5, 2, 3 },  { 2, 7, 3 },    
		{ 7, 10, 3 }, { 7, 6, 10 }, { 7, 11, 6 }, { 11, 0, 6 }, { 0, 1, 6 }, 
		{ 6, 1, 10 }, { 9, 0, 11 }, { 9, 11, 2 }, { 9, 2, 5 },  { 7, 2, 11 }
	};

	// Icosahedron has 20 faces, each subdivision turns a single triangle into four
	triangle_count = 20;
	for (int i = 0; i < num_subdivisions; i++) {
		triangle_count *= 4;
	}

	triangles = new Triangle[triangle_count];

	// Initialize with Icosahedron
	for (int i = 0; i < 20; i++) {
		triangles[i].position_0 = icosahedron_vertices[icosahedron_indices[i][0]];
		triangles[i].position_1 = icosahedron_vertices[icosahedron_indices[i][1]];
		triangles[i].position_2 = icosahedron_vertices[icosahedron_indices[i][2]];
	}

	// Subdivide N times
	int current_triangle_count = 20;
	for (int s = 0; s < num_subdivisions; s++) {
		for (int i = 0; i < current_triangle_count; i++) {
			Vector3 vertex_0 = triangles[i].position_0;
			Vector3 vertex_1 = triangles[i].position_1;
			Vector3 vertex_2 = triangles[i].position_2;

			Vector3 vertex_01 = Vector3::normalize(vertex_0 + vertex_1);
			Vector3 vertex_12 = Vector3::normalize(vertex_1 + vertex_2);
			Vector3 vertex_20 = Vector3::normalize(vertex_2 + vertex_0);

			triangles[i].position_0 = vertex_0;
			triangles[i].position_1 = vertex_01;
			triangles[i].position_2 = vertex_20;

			triangles[i + current_triangle_count].position_0 = vertex_01;
			triangles[i + current_triangle_count].position_1 = vertex_1;
			triangles[i + current_triangle_count].position_2 = vertex_12;

			triangles[i + 2 * current_triangle_count].position_0 = vertex_20;
			triangles[i + 2 * current_triangle_count].position_1 = vertex_12;
			triangles[i + 2 * current_triangle_count].position_2 = vertex_2;

			triangles[i + 3 * current_triangle_count].position_0 = vertex_01;
			triangles[i + 3 * current_triangle_count].position_1 = vertex_12;
			triangles[i + 3 * current_triangle_count].position_2 = vertex_20;
		}

		current_triangle_count *= 4;
	}

	// Bring into world space and calculate AABBs
	for (int i = 0; i < triangle_count; i++) {
		// Compute normals while positions are still centered around the origin
		triangles[i].normal_0 = Vector3::normalize(Matrix4::transform_direction(transform, triangles[i].position_0));
		triangles[i].normal_1 = Vector3::normalize(Matrix4::transform_direction(transform, triangles[i].position_1));
		triangles[i].normal_2 = Vector3::normalize(Matrix4::transform_direction(transform, triangles[i].position_2));

		triangles[i].position_0 = Matrix4::transform_position(transform, triangles[i].position_0);
		triangles[i].position_1 = Matrix4::transform_position(transform, triangles[i].position_1);
		triangles[i].position_2 = Matrix4::transform_position(transform, triangles[i].position_2);

		triangles[i].tex_coord_0 = Vector2(
			0.5f + atan2f(-triangles[i].normal_0.z, -triangles[i].normal_0.x) * ONE_OVER_TWO_PI,
			0.5f + asinf (-triangles[i].normal_0.y)                           * ONE_OVER_PI
		);
		triangles[i].tex_coord_1 = Vector2(
			0.5f + atan2f(-triangles[i].normal_1.z, -triangles[i].normal_1.x) * ONE_OVER_TWO_PI,
			0.5f + asinf (-triangles[i].normal_1.y)                           * ONE_OVER_PI
		);
		triangles[i].tex_coord_2 = Vector2(
			0.5f + atan2f(-triangles[i].normal_2.z, -triangles[i].normal_2.x) * ONE_OVER_TWO_PI,
			0.5f + asinf (-triangles[i].normal_2.y)                           * ONE_OVER_PI
		);

		Vector3 vertices[3] = {
			triangles[i].position_0,
			triangles[i].position_1,
			triangles[i].position_2
		};
		triangles[i].aabb = AABB::from_points(vertices, 3);
	}
}
