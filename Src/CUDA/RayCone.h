#pragma once
#include "Config.h"
#include "Triangle.h"

union TextureLOD {
	struct {
		float2 gradient_1;
		float2 gradient_2;
	} aniso;
	struct {
		float lod;
	} iso;
};

__device__ inline bool use_anisotropic_texture_sampling(int bounce) {
	return bounce == 0; // Only first bounce uses anisotropic sampling, subsequent bounces use isotropic
}

__device__ inline float3 sample_albedo(int bounce, float3 diffuse, int texture_id, float2 tex_coord, const TextureLOD & lod) {
	if (config.enable_mipmapping && texture_id != INVALID) {
		if (use_anisotropic_texture_sampling(bounce)) {
			return material_get_albedo(diffuse, texture_id, tex_coord.x, tex_coord.y, lod.aniso.gradient_1, lod.aniso.gradient_2);
		} else {
			return material_get_albedo(diffuse, texture_id, tex_coord.x, tex_coord.y, lod.iso.lod + textures[texture_id].lod_bias);
		}
	} else {
		return material_get_albedo(diffuse, texture_id, tex_coord.x, tex_coord.y);
	}
}

// Project the Ray Cone onto the Triangle and obtain two axes that describe the resulting ellipse (in world space)
__device__ inline void ray_cone_get_ellipse_axes(
	float3 ray_direction,
	float3 geometric_normal,
	float cone_width,
	float3 & ellipse_axis_1,
	float3 & ellipse_axis_2
) {
	float3 h_1 = ray_direction - dot(geometric_normal, ray_direction) * geometric_normal;
	float3 h_2 = cross(geometric_normal, h_1);

	ellipse_axis_1 = cone_width / max(0.0001f, length(h_1 - dot(ray_direction, h_1) * ray_direction)) * h_1;
	ellipse_axis_2 = cone_width / max(0.0001f, length(h_2 - dot(ray_direction, h_2) * ray_direction)) * h_2;
}

// Convert an ellipse axis into texture space
__device__ inline float2 ray_cone_ellipse_axis_to_gradient(
	const TrianglePosNorTex & triangle,
	float                     triangle_double_area_inv,
	const float3            & geometric_normal,
	const float3            & hit_point,
	const float2            & hit_tex_coord,
	const float3            & ellipse_axis
) {
	float3 e_p = hit_point + ellipse_axis - triangle.position_0;

	float u = dot(geometric_normal, cross(e_p, triangle.position_edge_2)) * triangle_double_area_inv;
	float v = dot(geometric_normal, cross(triangle.position_edge_1, e_p)) * triangle_double_area_inv;

	return barycentric(u, v, triangle.tex_coord_0, triangle.tex_coord_edge_1, triangle.tex_coord_edge_2) - hit_tex_coord;
}

__device__ inline float ray_cone_get_lod(float3 ray_direction, float3 geometric_normal, float cone_width) {
	return fabsf(cone_width / dot(ray_direction, geometric_normal));
}
