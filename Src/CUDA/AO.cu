#include "cudart/vector_types.h"
#include "cudart/cuda_math.h"

#include "Sampling.h"
#include "Buffers.h"
#include "Camera.h"

#include "Raytracing/BVH2.h"
#include "Raytracing/BVH4.h"
#include "Raytracing/BVH8.h"

__device__ __constant__ float4 * frame_buffer_ambient;

// Final Frame Buffer, shared with OpenGL
__device__ __constant__ Surface<float4> accumulator;

__device__ __constant__ Camera camera;

__device__ PixelQuery pixel_query = { INVALID, INVALID, INVALID };

// Input to the Trace and Ambient Occlusion Kernels in SoA layout
struct TraceBufferAO {
	TraversalData traversal_data;

	int * pixel_index;
};

// Input to the Shadow Trace Kernels in SoA layout
struct ShadowRayBufferAO {
	ShadowTraversalData traversal_data;

	int * pixel_index;
};

// Number of elements in each Buffer
struct BufferSizesAO {
	int trace;
	int shadow;

	// Global counters for tracing kernels
	int rays_retired;
	int rays_retired_shadow;
};

__device__ __constant__ TraceBufferAO     ray_buffer_trace;
__device__ __constant__ ShadowRayBufferAO ray_buffer_shadow;

__device__ BufferSizesAO buffer_sizes;

extern "C" __global__ void kernel_generate(int sample_index, int pixel_offset, int pixel_count) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= pixel_count) return;

	int index_offset = index + pixel_offset;
	int x = index_offset % screen_width;
	int y = index_offset / screen_width;

	int pixel_index = x + y * screen_pitch;

	Ray ray = camera_generate_ray(pixel_index, sample_index, x, y, camera);

	ray_buffer_trace.traversal_data.ray_origin   .set(index, ray.origin);
	ray_buffer_trace.traversal_data.ray_direction.set(index, ray.direction);

	ray_buffer_trace.pixel_index[index] = pixel_index;
}

extern "C" __global__ void kernel_trace_bvh2() {
	bvh2_trace(&ray_buffer_trace.traversal_data, buffer_sizes.trace, &buffer_sizes.rays_retired);
}

extern "C" __global__ void kernel_trace_bvh4() {
	bvh4_trace(&ray_buffer_trace.traversal_data, buffer_sizes.trace, &buffer_sizes.rays_retired);
}

extern "C" __global__ void kernel_trace_bvh8() {
	bvh8_trace(&ray_buffer_trace.traversal_data, buffer_sizes.trace, &buffer_sizes.rays_retired);
}

extern "C" __global__ void kernel_trace_shadow_bvh2() {
	bvh2_trace_shadow(&ray_buffer_shadow.traversal_data, buffer_sizes.shadow, &buffer_sizes.rays_retired_shadow, [](int ray_index) {
		int pixel_index = ray_buffer_shadow.pixel_index[ray_index];
		frame_buffer_ambient[pixel_index] = make_float4(1.0f);
	});
}

extern "C" __global__ void kernel_trace_shadow_bvh4() {
	bvh4_trace_shadow(&ray_buffer_shadow.traversal_data, buffer_sizes.shadow, &buffer_sizes.rays_retired_shadow, [](int ray_index) {
		int pixel_index = ray_buffer_shadow.pixel_index[ray_index];
		frame_buffer_ambient[pixel_index] = make_float4(1.0f);
	});
}

extern "C" __global__ void kernel_trace_shadow_bvh8() {
	bvh8_trace_shadow(&ray_buffer_shadow.traversal_data, buffer_sizes.shadow, &buffer_sizes.rays_retired_shadow, [](int ray_index) {
		int pixel_index = ray_buffer_shadow.pixel_index[ray_index];
		frame_buffer_ambient[pixel_index] = make_float4(1.0f);
	});
}

extern "C" __global__ void kernel_ambient_occlusion(int sample_index, float ao_radius) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= buffer_sizes.trace) return;

	float3 ray_direction = ray_buffer_trace.traversal_data.ray_direction.get(index);
	RayHit hit           = ray_buffer_trace.traversal_data.hits         .get(index);
	
	int pixel_index = ray_buffer_trace.pixel_index[index];

	if (hit.triangle_id == INVALID) {
		return; // Hit nothing
	}

	if (pixel_query.pixel_index == pixel_index) {
		pixel_query.mesh_id     = hit.mesh_id;
		pixel_query.triangle_id = hit.triangle_id;
	}

	// Obtain hit point and normal
	TrianglePosNor hit_triangle = triangle_get_positions_and_normals(hit.triangle_id);

	float3 geometric_normal = normalize(cross(hit_triangle.position_edge_1, hit_triangle.position_edge_2));

	float3 hit_point;
	float3 hit_normal;
	triangle_barycentric(hit_triangle, hit.u, hit.v, hit_point, hit_normal);

	// Transform into world space
	Matrix3x4 world = mesh_get_transform(hit.mesh_id);
	matrix3x4_transform_position (world, hit_point);
	matrix3x4_transform_direction(world, hit_normal);

	hit_normal = normalize(hit_normal);

	// Make sure the normal is always pointing outwards
	if (dot(ray_direction, hit_normal) > 0.0f) {
		hit_normal = -hit_normal;
	}

	// Construct TBN frame
	float3 tangent, bitangent;
	orthonormal_basis(hit_normal, tangent, bitangent);

	// Sample cosine weighted direction
	float2 rand_brdf = random<SampleDimension::BRDF>(pixel_index, 0, sample_index);
	float3 omega_o = sample_cosine_weighted_direction(rand_brdf.x, rand_brdf.y);

	float3 direction_out = local_to_world(omega_o, tangent, bitangent, hit_normal);
	float  pdf = omega_o.z * ONE_OVER_PI;

	if (!pdf_is_valid(pdf)) return;
	
	// Emit Shadow Ray
	int shadow_ray_index = atomicAdd(&buffer_sizes.shadow, 1);

	ray_buffer_shadow.traversal_data.ray_origin   .set(shadow_ray_index, ray_origin_epsilon_offset(hit_point, direction_out, geometric_normal));
	ray_buffer_shadow.traversal_data.ray_direction.set(shadow_ray_index, direction_out);
	ray_buffer_shadow.traversal_data.max_distance[shadow_ray_index] = ao_radius;
	ray_buffer_shadow.pixel_index[shadow_ray_index] = pixel_index;
}

extern "C" __global__ void kernel_accumulate(float frames_accumulated) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= screen_width || y >= screen_height) return;

	int pixel_index = x + y * screen_pitch;

	float4 colour = frame_buffer_ambient[pixel_index];

	if (frames_accumulated > 0.0f) {
		float4 colour_prev = accumulator.get(x, y);

		colour = colour_prev + (colour - colour_prev) / frames_accumulated; // Online average
	}

	if (!isfinite(colour.x + colour.y + colour.z)) {
		printf("WARNING: pixel (%i, %i) has colour (%f, %f, %f)!\n", x, y, colour.x, colour.y, colour.z);
		colour = make_float4(1000.0f, 0.0f, 1000.0f, 1.0f);
	}

	accumulator.set(x, y, colour);
}
