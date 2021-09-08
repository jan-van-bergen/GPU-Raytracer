#pragma once
#include "Math/Matrix4.h"

#include "Pathtracer/Triangle.h"

namespace Geometry {
	void rectangle(Triangle *& triangles, int & triangle_count, const Matrix4 & transform);
	void cube     (Triangle *& triangles, int & triangle_count, const Matrix4 & transform);
	void disk     (Triangle *& triangles, int & triangle_count, const Matrix4 & transform, int num_segments = 32);
	void cylinder (Triangle *& triangles, int & triangle_count, const Matrix4 & transform, const Vector3 & p0 = Vector3(0.0f), const Vector3 & p1 = Vector3(0.0f, 0.0f, 1.0f), float radius = 1.0f, int num_segments = 32);
	void sphere   (Triangle *& triangles, int & triangle_count, const Matrix4 & transform, int num_subdivisions = 2);
}
