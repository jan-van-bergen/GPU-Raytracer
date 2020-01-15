#include "vector_types.h"
#include "cuda_math.h"

#include <corecrt_math.h>

#include "../Common.h"

#define ASSERT(proposition, fmt, ...) { if (!(proposition)) printf(fmt, __VA_ARGS__); assert(proposition); }

surface<void, 2> frame_buffer;

#define USE_IMPORTANCE_SAMPLING true

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

	__device__ bool is_light() const {
		return dot(emittance, emittance) > 0.0f;
	}
};

__device__ int     light_count;
__device__ int   * light_indices;
__device__ float * light_areas;
__device__ float total_light_area;

struct Ray {
	float3 origin;
	float3 direction;
	float3 direction_inv;
};

struct RayHit {
	float distance = INFINITY;
	
	int triangle_id;
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
	float3 position0;
	float3 position_edge1;
	float3 position_edge2;

	float3 normal0;
	float3 normal_edge1;
	float3 normal_edge2; 
	
	float2 tex_coord0;
	float2 tex_coord_edge1;
	float2 tex_coord_edge2;

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

__device__ void triangle_trace(const Triangle & triangle, const Ray & ray, RayHit & ray_hit, int triangle_id) {
	float3 h = cross(ray.direction, triangle.position_edge2);
	float  a = dot(triangle.position_edge1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - triangle.position0;
	float  u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) return;

	float3 q = cross(s, triangle.position_edge1);
	float  v = f * dot(ray.direction, q);

	if (v < 0.0f || u + v > 1.0f) return;

	float t = f * dot(triangle.position_edge2, q);

	if (t < EPSILON || t >= ray_hit.distance) return;

	ray_hit.distance = t;

	ray_hit.triangle_id = triangle_id;
	ray_hit.material_id = triangle.material_id;

	ray_hit.point = ray.origin + t * ray.direction;
	ray_hit.normal = normalize(triangle.normal0 
		+ u * triangle.normal_edge1
		+ v * triangle.normal_edge2
	);
	ray_hit.uv = triangle.tex_coord0 
		+ u * triangle.tex_coord_edge1 
		+ v * triangle.tex_coord_edge2;
}

__device__ bool triangle_intersect(const Triangle & triangle, const Ray & ray, float max_distance) {
	float3 h = cross(ray.direction, triangle.position_edge2);
	float  a = dot(triangle.position_edge1, h);

	float  f = 1.0f / a;
	float3 s = ray.origin - triangle.position0;
	float  u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) return false;

	float3 q = cross(s, triangle.position_edge1);
	float  v = f * dot(ray.direction, q);

	if (v < 0.0f || u + v > 1.0f) return false;

	float t = f * dot(triangle.position_edge2, q);

	if (t < EPSILON || t >= max_distance) return false;

	return true;
}

__device__ void bvh_trace(const Ray & ray, RayHit & ray_hit) {
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
					triangle_trace(triangles[i], ray, ray_hit, i);
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

__device__ bool bvh_intersect(const Ray & ray, float max_distance) {
	int stack[64];
	int stack_size = 1;

	// Push root on stack
	stack[0] = 0;

	while (stack_size > 0) {
		// Pop Node of the stack
		const BVHNode & node = bvh_nodes[stack[--stack_size]];

		if (node.aabb.intersects(ray, max_distance)) {
			if (node.is_leaf()) {
				for (int i = node.first; i < node.first + node.count; i++) {
					if (triangle_intersect(triangles[i], ray, max_distance)) {
						return true;
					}
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

	return false;
}

__device__ int      sky_size;
__device__ float3 * sky_data;

__device__ float3 sample_sky(const float3 & direction) {
	// Formulas as described on https://www.pauldebevec.com/Probes/
    float r = 0.5f * ONE_OVER_PI * acos(direction.z) * rsqrt(direction.x*direction.x + direction.y*direction.y);

	float u = direction.x * r + 0.5f;
	float v = direction.y * r + 0.5f;

	// Convert to pixel coordinates
	int x = int(u * sky_size);
	int y = int(v * sky_size);

	int index = x + y * sky_size;
	index = max(index, 0);
	index = min(index, sky_size * sky_size);

	return sky_data[index];
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
	float  inv_length = rsqrt(length_squared);
	float3 random_point_on_unit_sphere = inv_length * direction;

	// If the point is on the wrong hemisphere, return its negative
	if (dot(normal, random_point_on_unit_sphere) < 0.0f) {
		return -random_point_on_unit_sphere;
	}

	return random_point_on_unit_sphere;
}

__device__ float3 cosine_weighted_diffuse_reflection(unsigned & seed, const float3 & normal) {
	float r0 = random_float(seed);
	float r1 = random_float(seed);

	float sin_theta, cos_theta;
	sincos(TWO_PI * r1, &sin_theta, &cos_theta);

	float r = sqrtf(r0);
	float x = r * cos_theta;
	float y = r * sin_theta;
	
	float3 direction = normalize(make_float3(x, y, sqrtf(1.0f - r0)));
	
	// Calculate a tangent vector from the normal vector
	float3 tangent;
	if (fabs(normal.x) > fabs(normal.y)) {
		tangent = make_float3(normal.z, 0.0f, -normal.x) * rsqrt(normal.x * normal.x + normal.z * normal.z);
	} else {
		tangent = make_float3(0.0f, -normal.z, normal.y) * rsqrt(normal.y * normal.y + normal.z * normal.z);
	}

	// The binormal is perpendicular to both the normal and tangent vectors
	float3 binormal = cross(normal, tangent);

	// Multiply the direction with the TBN matrix
	direction = normalize(make_float3(
		tangent.x * direction.x + binormal.x * direction.y + normal.x * direction.z, 
		tangent.y * direction.x + binormal.y * direction.y + normal.y * direction.z, 
		tangent.z * direction.x + binormal.z * direction.y + normal.z * direction.z
	));

	ASSERT(dot(direction, normal) > -1e-5, "Invalid dot: dot = %f, direction = (%f, %f, %f), normal = (%f, %f, %f)\n", 
		dot(direction, normal), direction.x, direction.y, direction.z, normal.x, normal.y, normal.z
	);

	return direction;
}

__device__ float3 sample(unsigned & seed, Ray & ray) {
	const int ITERATIONS = 10;
	
	float3 colour     = make_float3(0.0f);
	float3 throughput = make_float3(1.0f);
	
	bool last_specular = true;

	for (int bounce = 0; bounce < ITERATIONS; bounce++) {
		// Check ray against all triangles
		RayHit hit;
		bvh_trace(ray, hit);

		// Check if we didn't hit anything
		if (hit.distance == INFINITY) {
			return colour + throughput * sample_sky(ray.direction);
		}

		const Material & material = materials[hit.material_id];

		if (light_count > 0) {
			if (material.is_light()) {
				if (last_specular) {
					return colour + throughput * material.emittance;
				} else {
					return colour;
				}
			}

			// Pick a random light emitting triangle
			const Triangle & light_triangle = triangles[light_indices[rand_xorshift(seed) % light_count]];

			ASSERT(length(materials[light_triangle.material_id].emittance) > 0.0f, "Material was not emissive!\n");
		
			// Pick a random point on the triangle using random barycentric coordinates
			float u = random_float(seed);
			float v = random_float(seed);

			if (u + v > 1.0f) {
				u = 1.0f - u;
				v = 1.0f - v;
			}

			float3 random_point_on_light = light_triangle.position0 
				+ u * light_triangle.position_edge1 
				+ v * light_triangle.position_edge2;

			// Calculate the area of the triangle light
			float light_area = 0.5f * length(cross(light_triangle.position_edge1, light_triangle.position_edge2));

			float3 to_light = random_point_on_light - hit.point;
			float distance_to_light_squared = dot(to_light, to_light);
			float distance_to_light         = sqrtf(distance_to_light_squared);

			// Normalize the vector to the light
			to_light /= distance_to_light;

			float3 light_normal = light_triangle.normal0 
				+ u * light_triangle.normal_edge1
				+ v * light_triangle.normal_edge2;

			float cos_o = -dot(to_light, light_normal);
			float cos_i =  dot(to_light, hit.normal);

			if (cos_o > 0.0f && cos_i > 0.0f) {
				ray.origin    = hit.point;
				ray.direction = to_light;
				ray.direction_inv = make_float3(
					1.0f / ray.direction.x, 
					1.0f / ray.direction.y, 
					1.0f / ray.direction.z
				);

				// Check if the light is obstructed by any other object in the scene
				if (!bvh_intersect(ray, distance_to_light - EPSILON)) {
					float3 brdf = material.albedo(hit.uv.x, hit.uv.y) * ONE_OVER_PI;
					float solid_angle = (cos_o * light_area) / distance_to_light_squared;

					float3 light_colour = materials[light_triangle.material_id].emittance;

					colour += throughput * brdf * light_count * light_colour * solid_angle * cos_i;
				}
			}
		}

#if USE_IMPORTANCE_SAMPLING
		float3 direction = cosine_weighted_diffuse_reflection(seed, hit.normal);

		throughput *= material.albedo(hit.uv.x, hit.uv.y);
#else
		float3 direction = diffuse_reflection(seed, hit.normal);

		throughput *= 2.0f * material.albedo(hit.uv.x, hit.uv.y) * dot(hit.normal, direction);
#endif

		// Russian Roulette termination after at least four bounces
		if (bounce > 3) {
			float one_minus_p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
			if (random_float(seed) > one_minus_p) {
				return colour;
			}

			throughput /= one_minus_p;
		}

		ray.origin    = hit.point;
		ray.direction = direction;
		ray.direction_inv = make_float3(
			1.0f / ray.direction.x, 
			1.0f / ray.direction.y, 
			1.0f / ray.direction.z
		);

		last_specular = false;
	}

	return make_float3(0.0f);
}

extern "C" __global__ void trace_ray(int frame_number, float frames_since_camera_moved) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int thread_id = x + y * SCREEN_WIDTH;

	unsigned seed = (thread_id + frame_number * 312080213) * 781939187;
	
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
	if (frames_since_camera_moved > 0.0f) {
		float4 prev;
		surf2Dread<float4>(&prev, frame_buffer, x * sizeof(float4), y);

		// Take average over n samples by weighing the current content of the framebuffer by (n-1) and the new sample by 1
		colour = (make_float3(prev) * (frames_since_camera_moved - 1.0f) + colour) / frames_since_camera_moved;
	}

	surf2Dwrite<float4>(make_float4(colour, 1.0f), frame_buffer, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

//////////////////////////////////////////////////////////////////////////////////////////

#define MAX_BOUNCES 10

// Based on: https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ inline int atomic_agg_inc(int * ctr) {
	int mask   = __ballot(1);
	int leader = __ffs(mask) - 1;
	int laneid = threadIdx.x % 32;
	
	int res;
	if (laneid == leader) {
		res = atomicAdd(ctr, __popc(mask));
	}

	res = __shfl(res, leader);
	return res + __popc(mask & ((1 << laneid) - 1));
}

__device__ void frame_buffer_write(int x, int y, const float3 & colour, float frames_since_camera_moved) {
	float3 colour_out = colour;

	if (frames_since_camera_moved > 0.0f) {
		float4 prev;
		surf2Dread<float4>(&prev, frame_buffer, x * sizeof(float4), y);

		// Take average over n samples by weighing the current content of the framebuffer by (n-1) and the new sample by 1
		colour_out = (make_float3(prev) * (frames_since_camera_moved - 1.0f) + colour) / frames_since_camera_moved;
	}

	surf2Dwrite<float4>(make_float4(colour_out, 1.0f), frame_buffer, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

struct WFRay {
	float3 origin;
	float3 direction;
	
	int triangle_id;
	float u, v;
	float t;

	float3 throughput;
	int pixel_index;
};

__device__ WFRay * buffer_rays_0;
__device__ WFRay * buffer_rays_1;

extern "C" __global__ void kernel_generate(
	int rand_seed,
	int buffer_size,
	float3 camera_position,
	float3 camera_top_left_corner,
	float3 camera_x_axis,
	float3 camera_y_axis
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_size) return;

	int x = index % SCREEN_WIDTH;
	int y = index / SCREEN_WIDTH;

	int thread_id = x + y * SCREEN_WIDTH;
	unsigned seed = (thread_id + rand_seed * 199494991) * 949525949;
	
	// Add random value between 0 and 1 so that after averaging we get anti-aliasing
	float u = x + random_float(seed);
	float v = y + random_float(seed);

	// Create primary Ray that starts at the Camera's position and goes trough the current pixel
	buffer_rays_0[index].origin    = camera_position;
	buffer_rays_0[index].direction = normalize(camera_top_left_corner
		+ u * camera_x_axis
		+ v * camera_y_axis
	);
	buffer_rays_0[index].throughput  = make_float3(1.0f);
	buffer_rays_0[index].pixel_index = thread_id;
}

extern "C" __global__ void kernel_extend(
	int buffer_size,
	WFRay * buffer_rays
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_size) return;

	Ray ray;
	ray.origin    = buffer_rays[index].origin;
	ray.direction = buffer_rays[index].direction;
	ray.direction_inv = make_float3(
		1.0f / ray.direction.x, 
		1.0f / ray.direction.y, 
		1.0f / ray.direction.z
	);

	RayHit hit;
	bvh_trace(ray, hit);

	if (hit.distance == INFINITY) {
		buffer_rays[index].triangle_id = -1;

		return;
	}

	buffer_rays[index].triangle_id = hit.triangle_id;
	buffer_rays[index].u = hit.uv.x;
	buffer_rays[index].v = hit.uv.y;
	buffer_rays[index].t = hit.distance;
}

__device__ int N_ext;

extern "C" __global__ void kernel_shade(
	int rand_seed,
	int buffer_size,
	int bounce,
	float frames_since_camera_moved,
	const WFRay * __restrict__ buffer_rays_in,
	      WFRay * __restrict__ buffer_rays_out
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_size) return;

	const WFRay & ray = buffer_rays_in[index];

	int x = ray.pixel_index % SCREEN_WIDTH;
	int y = ray.pixel_index / SCREEN_WIDTH; 

	// If the Ray didn't hit a Triangle, terminate the Path
	if (ray.triangle_id == -1) {
		frame_buffer_write(x, y, ray.throughput * sample_sky(ray.direction), frames_since_camera_moved);

		return;
	}

	unsigned seed = (ray.pixel_index + rand_seed * 312080213) * 781939187;

	const Triangle & triangle = triangles[ray.triangle_id];
	const Material & material = materials[triangle.material_id];

	if (material.is_light()) {
		frame_buffer_write(x, y, ray.throughput * material.emittance, frames_since_camera_moved);

		return;
	}

	float3 throughput = ray.throughput;

	// Russian Roulette termination
	if (bounce > 3) {
		float one_minus_p = fmaxf(throughput.x, fmaxf(throughput.y, throughput.z));
		if (random_float(seed) > one_minus_p) {
			return;
		}

		throughput /= one_minus_p;
	}

	int index_out = atomic_agg_inc(&N_ext);

	buffer_rays_out[index_out].origin    = ray.origin + ray.t * ray.direction;
	buffer_rays_out[index_out].direction = cosine_weighted_diffuse_reflection(seed, triangle.normal0
		+ ray.u * triangle.normal_edge1
		+ ray.v * triangle.normal_edge2
	);

	buffer_rays_out[index_out].throughput  = throughput * material.albedo(ray.u, ray.v);
	buffer_rays_out[index_out].pixel_index = ray.pixel_index;
}

extern "C" __global__ void kernel_connect() {
	
}
