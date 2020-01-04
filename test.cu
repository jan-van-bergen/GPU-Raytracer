#include "vector_types.h"
#include "cuda_math.h"

#include <corecrt_math.h>

#include "Common.h"

surface<void, 2> frame_buffer;

struct Material {
	float3 diffuse;
	int texture_id;
};

__device__ Material            * materials;
__device__ cudaTextureObject_t * textures;

struct Ray {
	float3 origin;
	float3 direction;
};

struct RayHit {
	float distance = INFINITY;
	
	int material_id;

	float3 point;
	float3 normal;
	float2 uv;
};

struct Triangle {
	float3 position0;
	float3 position1;
	float3 position2;

	float3 normal0;
	float3 normal1;
	float3 normal2; 
	
	float2 tex_coord0;
	float2 tex_coord1;
	float2 tex_coord2;

	int material_id;
};

__device__ int        triangle_count;
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

	if (u < -EPSILON || u > 1.0f + EPSILON) return;

	float3 q = cross(s, edge1);
	float  v = f * dot(ray.direction, q);

	if (v < -EPSILON || u + v > 1.0f + EPSILON) return;

	float t = f * dot(edge2, q);

	if (t < EPSILON || t >= ray_hit.distance) return;

	ray_hit.distance = t;

	ray_hit.material_id = triangle.material_id;

	ray_hit.point = ray.origin + t * ray.direction;
	ray_hit.normal = normalize(triangle.normal0 
		+ u * (triangle.normal1 - triangle.normal0) 
		+ v * (triangle.normal2 - triangle.normal0)
	);
	ray_hit.uv = triangle.tex_coord0 
		+ u * (triangle.tex_coord1 - triangle.tex_coord0) 
		+ v * (triangle.tex_coord2 - triangle.tex_coord0);
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
		assert(hit.material_id >= 0 && hit.material_id < MAX_MATERIALS);
		
		int texture_id = materials[hit.material_id].texture_id;

		if (texture_id == -1) {
			colour = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
		} else {
			float4 tex_colour;

			for (int i = 0; i < MAX_TEXTURES; i++) {
				if (texture_id == i) {
					tex_colour = tex2D<float4>(textures[i], hit.uv.x, hit.uv.y);
				}
			}

			colour = make_float4(tex_colour.x, tex_colour.y, tex_colour.z, 1.0f);
		}
	}

	surf2Dwrite<float4>(colour, frame_buffer, x * sizeof(float4), y, cudaBoundaryModeClamp);
}
