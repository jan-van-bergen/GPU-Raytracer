#include "Triangle.h"

#include <mikktspace/mikktspace.h>

static Array<Triangle> & get_triangles(const SMikkTSpaceContext * context) {
	return *reinterpret_cast<Array<Triangle> *>(context->m_pUserData);
}

static int get_num_faces(const SMikkTSpaceContext * context) {
	return get_triangles(context).size();
}

static int get_num_vertices_of_face(const SMikkTSpaceContext * context, int face_index) {
	return 3;
}

static void get_position(const SMikkTSpaceContext * context, float position[], int face_index, int vertex_index) {
	Vector3 * p = nullptr;
	switch (vertex_index) {
		case 0: p = &get_triangles(context)[face_index].position_0; break;
		case 1: p = &get_triangles(context)[face_index].position_1; break;
		case 2: p = &get_triangles(context)[face_index].position_2; break;
	}

	position[0] = p->x;
	position[1] = p->y;
	position[2] = p->z;
}

static void get_normal(const SMikkTSpaceContext * context, float normal[], int face_index, int vertex_index) {
	Vector3 * n = nullptr;
	switch (vertex_index) {
		case 0: n = &get_triangles(context)[face_index].normal_0; break;
		case 1: n = &get_triangles(context)[face_index].normal_1; break;
		case 2: n = &get_triangles(context)[face_index].normal_2; break;
	}

	normal[0] = n->x;
	normal[1] = n->y;
	normal[2] = n->z;
}

static void get_tex_coord(const SMikkTSpaceContext * context, float tex_coord[], int face_index, int vertex_index) {
	Vector2 * t = nullptr;
	switch (vertex_index) {
		case 0: t = &get_triangles(context)[face_index].tex_coord_0; break;
		case 1: t = &get_triangles(context)[face_index].tex_coord_1; break;
		case 2: t = &get_triangles(context)[face_index].tex_coord_2; break;
	}

	tex_coord[0] = t->x;
	tex_coord[1] = t->y;
}

static void set_tangent(const SMikkTSpaceContext * context, const float tangent[], float sign, int face_index, int vertex_index) {
	switch (vertex_index) {
		case 0: get_triangles(context)[face_index].tangent_0 = sign * Vector3(tangent); break;
		case 1: get_triangles(context)[face_index].tangent_1 = sign * Vector3(tangent); break;
		case 2: get_triangles(context)[face_index].tangent_2 = sign * Vector3(tangent); break;
	}
}

void Triangle::generate_tangents(Array<Triangle> & triangles) {
	SMikkTSpaceInterface interface = { };
	interface.m_getNumFaces          = &get_num_faces;
	interface.m_getNumVerticesOfFace = &get_num_vertices_of_face;
	interface.m_getPosition          = &get_position;
	interface.m_getNormal            = &get_normal;
	interface.m_getTexCoord          = &get_tex_coord;
	interface.m_setTSpaceBasic       = &set_tangent;

	SMikkTSpaceContext context = { };
	context.m_pInterface = &interface;
	context.m_pUserData  = &triangles;

	tbool success = genTangSpaceDefault(&context);
	ASSERT(success);
}
