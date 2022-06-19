#pragma once
#include "Config.h"

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

__device__ inline float3 sample_albedo(int bounce, const float3 & diffuse, int texture_id, float2 tex_coord, const TextureLOD & lod) {
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

__device__ inline void ray_cone_get_ellipse_axes(
	const float3 & ray_direction,
	const float3 & geometric_normal,
	float cone_width,
	float3 & ellipse_axis_1,
	float3 & ellipse_axis_2
) {
	float3 h_1 = ray_direction - dot(geometric_normal, ray_direction) * geometric_normal;
	float3 h_2 = cross(geometric_normal, h_1);

	ellipse_axis_1 = cone_width / max(0.0001f, length(h_1 - dot(ray_direction, h_1) * ray_direction)) * h_1;
	ellipse_axis_2 = cone_width / max(0.0001f, length(h_2 - dot(ray_direction, h_2) * ray_direction)) * h_2;
}

__device__ inline void ray_cone_get_texture_gradients(
		  float    mesh_scale,
	const float3 & geometric_normal,
		  float    triangle_area_inv,
	const float3 & position_0,
	const float3 & position_edge_1,
	const float3 & position_edge_2,
	const float2 & tex_coord_0,
	const float2 & tex_coord_edge_1,
	const float2 & tex_coord_edge_2,
	const float3 & hit_point,
	const float2 & hit_tex_coord,
	const float3 & ellipse_axis_1,
	const float3 & ellipse_axis_2,
		  float2 & gradient_1,
		  float2 & gradient_2
) {
	float3 e_p = hit_point + ellipse_axis_1 - position_0;

	float inv_mesh_scale = 1.0f / mesh_scale;

	float u_1 = dot(geometric_normal, cross(e_p, position_edge_2)) * triangle_area_inv;
	float v_1 = dot(geometric_normal, cross(position_edge_1, e_p)) * triangle_area_inv;
	gradient_1 = inv_mesh_scale * (barycentric(u_1, v_1, tex_coord_0, tex_coord_edge_1, tex_coord_edge_2) - hit_tex_coord);

	e_p = hit_point + ellipse_axis_2 - position_0;

	float u_2 = dot(geometric_normal, cross(e_p, position_edge_2)) * triangle_area_inv;
	float v_2 = dot(geometric_normal, cross(position_edge_1, e_p)) * triangle_area_inv;
	gradient_2 = inv_mesh_scale * (barycentric(u_2, v_2, tex_coord_0, tex_coord_edge_1, tex_coord_edge_2) - hit_tex_coord);
}

__device__ inline float ray_cone_get_lod(const float3 & ray_direction, const float3 & geometric_normal, float cone_width) {
	return fabsf(cone_width / dot(ray_direction, geometric_normal));
}
