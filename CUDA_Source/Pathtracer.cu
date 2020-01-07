#include "vector_types.h"
#include "cuda_math.h"

#include <corecrt_math.h>

#include "../Common.h"

#define PI 3.14159265359f

surface<void, 2> frame_buffer;

// Based on: http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
__device__ unsigned wang_hash(unsigned seed) {
	seed = (seed ^ 61) ^ (seed >> 16);
	seed = seed + (seed << 3);
	seed = seed ^ (seed >> 4);
	seed = seed * 0x27d4eb2d;
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

struct Material;

__device__ Material            * materials;
__device__ cudaTextureObject_t * textures;

struct Material {
	float3 diffuse;
	int texture_id;

	float3 emittance;

	__device__ float3 albedo(float u, float v) const {
		if (texture_id == -1) return diffuse;

		float4 tex_colour;

		for (int i = 0; i < MAX_TEXTURES; i++) {
			if (texture_id == i) {
				tex_colour = tex2D<float4>(textures[i], u, v);
			}
		}

		return diffuse * make_float3(tex_colour);
	}
};

struct Ray {
	float3 origin;
	float3 direction;
	float3 direction_inv;
};

struct RayHit {
	float distance = INFINITY;
	
	int material_id;

	float3 point;
	float3 normal;
	float2 uv;
};

struct AABB {
	float3 min;
	float3 max;

	__device__ inline bool intersects(const Ray & ray, float max_distance) const {
		float3 t0 = (min - ray.origin) * ray.direction_inv;
		float3 t1 = (max - ray.origin) * ray.direction_inv;
		
		float3 t_min = fminf(t0, t1);
		float3 t_max = fmaxf(t0, t1);
		
		float t_near = fmaxf(fmaxf(EPSILON,      t_min.x), fmaxf(t_min.y, t_min.z));
		float t_far  = fminf(fminf(max_distance, t_max.x), fminf(t_max.y, t_max.z));
	
		return t_near < t_far;
	}
};

struct Triangle {
	AABB aabb;

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

__device__ Triangle * triangles;

struct BVHNode {
	AABB aabb;
	union {
		int left;
		int first;
	};
	int count;

	__device__ inline bool is_leaf() const {
		return (count & (~BVH_AXIS_MASK)) > 0;
	}

	__device__ inline bool should_visit_left_first(const Ray & ray) const {
#if BVH_TRAVERSAL_STRATEGY == BVH_TRAVERSE_TREE_NAIVE
		return true; // Naive always goes left first
#elif BVH_TRAVERSAL_STRATEGY == BVH_TRAVERSE_TREE_ORDERED
		switch (count & BVH_AXIS_MASK) {
			case BVH_AXIS_X_BITS: return ray.direction.x > 0.0f;
			case BVH_AXIS_Y_BITS: return ray.direction.y > 0.0f;
			case BVH_AXIS_Z_BITS: return ray.direction.z > 0.0f;
		}
#endif
	}
};

__device__ BVHNode * bvh_nodes;

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

__device__ void bvh_traverse(const Ray & ray, RayHit & ray_hit) {
	int stack[64];
	int stack_size = 1;

	// Push root on stack
	stack[0] = 0;

	while (stack_size > 0) {
		// Pop Node of the stack
		const BVHNode & node = bvh_nodes[stack[--stack_size]];

		if (node.aabb.intersects(ray, ray_hit.distance)) {
			if (node.is_leaf()) {
				for (int i = node.first; i < node.first + node.count; i++) {
					check_triangle(triangles[i], ray, ray_hit);
				}
			} else {
				if (node.should_visit_left_first(ray)) {
					stack[stack_size++] = node.left + 1;
					stack[stack_size++] = node.left;
				} else {
					stack[stack_size++] = node.left;
					stack[stack_size++] = node.left + 1;
				}
			}
		}
	}
}

__device__ float3 diffuse_reflection(unsigned & seed, const float3 & normal) {
	float3 direction;
	float  length_squared;

	// Find a random point inside the unit sphere
	do {
		direction.x = -1.0f + 2.0f * random_float(seed);
		direction.y = -1.0f + 2.0f * random_float(seed);
		direction.z = -1.0f + 2.0f * random_float(seed);

		length_squared = dot(direction, direction);
	} while (length_squared > 1.0f);

	// Normalize direction to obtain a random point on the unit sphere
	float  inv_length = 1.0f / sqrt(length_squared);
	float3 random_point_on_unit_sphere = inv_length * direction;

	// If the point is on the wrong hemisphere, return its negative
	if (dot(normal, random_point_on_unit_sphere) < 0.0f) {
		return -random_point_on_unit_sphere;
	}

	return random_point_on_unit_sphere;
}

__device__ float3 sample(unsigned & seed, Ray & ray) {
	const int ITERATIONS = 5;
	
	float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
	
	for (int i = 0; i < ITERATIONS; i++) {
		// Check ray against all triangles
		RayHit hit;
		bvh_traverse(ray, hit);

		// Check if we didn't hit anything
		if (hit.distance == INFINITY) {
			return make_float3(0.0f);
		}

		const Material & material = materials[hit.material_id];

		// Check if we hit a Light
		if (material.emittance.x > 0.0f || material.emittance.y > 0.0f || material.emittance.z > 0.0f) {
			return throughput * material.emittance;
		}

		// Create new Ray in random direction on the hemisphere defined by the normal
		float3 diffuse_reflection_direction = diffuse_reflection(seed, hit.normal);

		ray.origin    = hit.point;
		ray.direction = diffuse_reflection_direction;
		ray.direction_inv = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
		
		throughput *= 2.0f * material.albedo(hit.uv.x, hit.uv.y) * dot(hit.normal, diffuse_reflection_direction);
	}

	return make_float3(0.0f);
}

extern "C" __global__ void trace_ray(int random, float frames_since_last_camera_moved) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	unsigned seed = wang_hash(x*random + y*random * SCREEN_WIDTH);
	
	// Add random value between 0 and 1 so that after averaging we get anti-aliasing
	float u = x + random_float(seed);
	float v = y + random_float(seed);

	// Create primary Ray that starts at the Camera's position and goes trough the current pixel
	Ray ray;
	ray.origin    = camera_position;
	ray.direction = normalize(camera_top_left_corner
		+ u * camera_x_axis
		+ v * camera_y_axis
	);
	ray.direction_inv = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);
	
	float3 colour = sample(seed, ray);

	// If the Camera hasn't moved, average over previous frames
	if (frames_since_last_camera_moved > 0.0f) {
		float4 prev;
		surf2Dread<float4>(&prev, frame_buffer, x * sizeof(float4), y);

		// Take average over n samples by weighing the current content of the framebuffer by (n-1) and the new sample by 1
		colour = (make_float3(prev) * (frames_since_last_camera_moved - 1.0f) + colour) / frames_since_last_camera_moved;
	}

	surf2Dwrite<float4>(make_float4(colour, 1.0f), frame_buffer, x * sizeof(float4), y, cudaBoundaryModeClamp);
}
