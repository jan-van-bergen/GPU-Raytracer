__device__ inline void ray_cone_get_ellipse_axes(
	const float3 & ray_direction,
	const float3 & geometric_normal,
	float cone_width,
	float3 & axis_1,
	float3 & axis_2
) {
	float3 h_1 = ray_direction - dot(geometric_normal, ray_direction) * geometric_normal;
	float3 h_2 = cross(geometric_normal, h_1);
	
	axis_1 = cone_width / max(0.0001f, length(h_1 - dot(ray_direction, h_1) * ray_direction)) * h_1;
	axis_2 = cone_width / max(0.0001f, length(h_2 - dot(ray_direction, h_2) * ray_direction)) * h_2;
}

__device__ inline void ray_cone_get_texture_gradients(
	const float3 & geometric_normal,
	const float3 & position_0,
	const float3 & position_edge_1,
	const float3 & position_edge_2,
	const float2 & tex_coord_0,
	const float2 & tex_coord_edge_1,
	const float2 & tex_coord_edge_2,
	const float3 & hit_point,
	const float2 & hit_tex_coord,
	const float3 & a_1,
	const float3 & a_2,
	float2 & g_1,
	float2 & g_2
) {
	float triangle_area_inv = 1.0f / dot(geometric_normal, cross(position_edge_1, position_edge_2));

	float3 e_p = hit_point + a_1 - position_0;

	float u_1 = dot(geometric_normal, cross(e_p, position_edge_2)) * triangle_area_inv;
	float v_1 = dot(geometric_normal, cross(position_edge_1, e_p)) * triangle_area_inv;
	g_1 = barycentric(u_1, v_1, tex_coord_0, tex_coord_edge_1, tex_coord_edge_2) - hit_tex_coord;

	e_p = hit_point + a_2 - position_0;

	float u_2 = dot(geometric_normal, cross(e_p, position_edge_2)) * triangle_area_inv;
	float v_2 = dot(geometric_normal, cross(position_edge_1, e_p)) * triangle_area_inv;
	g_2 = barycentric(u_2, v_2, tex_coord_0, tex_coord_edge_1, tex_coord_edge_2) - hit_tex_coord;

}

__device__ inline float ray_cone_get_lod(const float3 & ray_direction, const float3 & geometric_normal, float cone_width) {
	return log2f(cone_width / fabsf(dot(ray_direction, geometric_normal)));
}
