#pragma once
#include "Core/Array.h"

#include "Math/Matrix4.h"
#include "Renderer/Triangle.h"

namespace Geometry {
	Array<Triangle> rectangle(const Matrix4 & transform);
	Array<Triangle> cube     (const Matrix4 & transform);
	Array<Triangle> disk     (const Matrix4 & transform, int num_segments = 32);
	Array<Triangle> cylinder (const Matrix4 & transform, const Vector3 & p0 = Vector3(0.0f), const Vector3 & p1 = Vector3(0.0f, 0.0f, 1.0f), float radius = 1.0f, int num_segments = 32);
	Array<Triangle> sphere   (const Matrix4 & transform, int num_subdivisions = 3);
}
