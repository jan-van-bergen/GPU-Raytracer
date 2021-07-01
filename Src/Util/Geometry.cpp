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
