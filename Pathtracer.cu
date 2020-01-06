#include "vector_types.h"
#include "cuda_math.h"

#include <corecrt_math.h>

#include "Common.h"

#define PI 3.14159265359f

surface<void, 2> frame_buffer;

// Based on: http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
__device__ unsigned wang_hash(unsigned seed) {
	// seed = (seed ^ 61) ^ (seed >> 16);
	// seed = seed + (seed << 3);
	// seed = seed ^ (seed >> 4);
	// seed = seed * 0x27d4eb2d;
	// seed = seed ^ (seed >> 15);

	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);

	return seed;
}

// Based on: http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
__device__ unsigned rand_xorshift(unsigned & seed) {
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
	seed ^= (seed << 5);
	
    return seed;
}

__device__ float random_float(unsigned & seed) {
	const float one_over_max_unsigned = 2.3283064365387e-10f;
	return float(rand_xorshift(seed)) * one_over_max_unsigned;
}

struct Material {
	float3 diffuse;
	int texture_id;

	float3 emittance;
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


__device__ float3 diffuse_reflection(unsigned & seed, const float3 & normal) {
	// @TODO: use faster method to do this!

	// Generate random spherical coordinates
	float theta = 2.0f * acos(sqrt(1.0f - random_float(seed)));
	float phi   = 2.0f * PI * random_float(seed);

	float sin_theta, cos_theta;
	float sin_phi,   cos_phi;

	sincos(theta, &sin_theta, &cos_theta);
	sincos(phi,   &sin_phi,   &cos_phi);

	// Convert spherical coordinates to cartesian coordinates
	float3 random_point_on_unit_sphere = make_float3(
		sin_theta * cos_phi, 
		sin_theta * sin_phi, 
		cos_theta
	);

	// If the point is on the wrong hemisphere, return its negative
	if (dot(normal, random_point_on_unit_sphere) < 0.0f) {
		return -random_point_on_unit_sphere;
	}

	return random_point_on_unit_sphere;
}

__device__ float3 sample(unsigned & seed, Ray & ray) {
	const int ITERATIONS = 10;

	float3 throughput = make_float3(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < ITERATIONS; i++) {
		// Check ray against all triangles
		RayHit hit;
		for (int i = 0; i < triangle_count; i++) {
			check_triangle(triangles[i], ray, hit);
		}

		// Check if we didn't hit anything
		if (hit.distance == INFINITY) {
			throughput = make_float3(0.0f, 0.0f, 0.0f);

			break;
		}

		const Material & material = materials[hit.material_id];

		// Check if we hit a Light
		if (material.emittance.x > 0.0f || material.emittance.y > 0.0f || material.emittance.z > 0.0f) {
			throughput *= material.emittance;

			break;
		}

		// Create new Ray in random direction on the hemisphere defined by the normal
		float3 diffuse_reflection_direction = diffuse_reflection(seed, hit.normal);

		ray.origin    = hit.point;
		ray.direction = diffuse_reflection_direction;

		// int texture_id = materials[hit.material_id].texture_id;

		// if (texture_id == -1) {
		// 	colour = make_float4(1.0f, 0.0f, 1.0f, 1.0f);
		// } else {
		// 	float4 tex_colour;

		// 	for (int i = 0; i < MAX_TEXTURES; i++) {
		// 		if (texture_id == i) {
		// 			tex_colour = tex2D<float4>(textures[i], hit.uv.x, hit.uv.y);
		// 		}
		// 	}

		// 	colour = make_float4(tex_colour.x, tex_colour.y, tex_colour.z, 1.0f);
		// }

		throughput *= 2.0f * material.diffuse * dot(hit.normal, diffuse_reflection_direction);
	}

	return throughput;
}

extern "C" __global__ void trace_ray(int random, float frames_since_last_camera_moved) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	Ray ray;
	ray.origin    = camera_position;
	ray.direction = normalize(camera_top_left_corner
		+ x * camera_x_axis
		+ y * camera_y_axis
	);

	int thread_id = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
	unsigned seed = wang_hash(thread_id + random);
	float3 colour = sample(seed, ray);
	
	if (frames_since_last_camera_moved > 0.0f) {
		float4 prev;
		surf2Dread<float4>(&prev, frame_buffer, x * sizeof(float4), y);

		float3 old_average = make_float3(prev.x, prev.y, prev.z);

		colour = (old_average * (frames_since_last_camera_moved - 1.0f) + colour) / frames_since_last_camera_moved;
	}

	surf2Dwrite<float4>(make_float4(colour.x, colour.y, colour.z, 1.0f), frame_buffer, x * sizeof(float4), y, cudaBoundaryModeClamp);
}
