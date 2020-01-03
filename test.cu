#include "vector_types.h"
#include "cuda_math.h"

#include <corecrt_math.h>

#include "Common.h"

surface<void, 2> output_surface;

struct Ray {
	float3 origin;
	float3 direction;
};

struct RayHit {
	float distance = INFINITY;
	
	float3 point;
	float3 normal;
};

struct Triangle {
	float3 position0;
	float3 position1;
	float3 position2;
};

__device__ int triangle_count;
__device__ Triangle * triangles;

__device__ float3 camera_position;
__device__ float3 camera_top_left_corner;
__device__ float3 camera_x_axis;
__device__ float3 camera_y_axis;

__device__ float3 get_direction(float x, float y) {
	return normalize(camera_top_left_corner
		+ x * camera_x_axis
		+ y * camera_y_axis
	);
}

__device__ void check_triangle(const Triangle & triangle, const Ray & ray, RayHit & ray_hit) {
	float3 edge1 = triangle.position1 - triangle.position0;
	float3 edge2 = triangle.position2 - triangle.position0;

	float3 h = cross(ray.direction, edge2);
	float  a = dot(edge1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - triangle.position0;
	float  u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) return;

	float3 q = cross(s, edge1);
	float  v = f * dot(ray.direction, q);

	if (v < 0.0f || u + v > 1.0f) return;

	float t = f * dot(edge2, q);

	if (t < EPSILON || t >= ray_hit.distance) return;

	ray_hit.distance = t;

	ray_hit.point = ray.origin + t * ray.direction;
	ray_hit.normal = normalize(cross(edge1, edge2)); // @TODO: use barycentric interpolation on Vertex normals!
}

extern "C" __global__ void trace_ray() {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	Ray ray;
	ray.origin    = camera_position;
	ray.direction = get_direction(x, y);

	RayHit hit;
	
	for (int i = 0; i < triangle_count; i++) {
		check_triangle(triangles[i], ray, hit);
	}

	float4 colour = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	
	if (hit.distance < INFINITY) {
		colour = make_float4(hit.point.x, hit.point.y, hit.point.z, 1.0f);
	}

	surf2Dwrite<float4>(colour, output_surface, x * sizeof(float4), y, cudaBoundaryModeClamp);
}
