#pragma once
#include "Math/Matrix4.h"

#include "Pathtracer/Triangle.h"

namespace Geometry {
	void rectangle(Triangle *& triangles, int & triangle_count, const Matrix4 & transform);
	void cube     (Triangle *& triangles, int & triangle_count, const Matrix4 & transform);
	void disk     (Triangle *& triangles, int & triangle_count, const Matrix4 & transform, int num_segments = 16);
	void sphere   (Triangle *& triangles, int & triangle_count, const Matrix4 & transform, int num_subdivisions = 2);
}
