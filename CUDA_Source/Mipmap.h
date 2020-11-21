
#if ENABLE_MIPMAPPING
// Samples the given Material's albedo texture map at the appropriate LOD
// The LOD is determined using ray differentials. The equations use some
// simplifications that are only appropriate for primary rays.
// For subsequent bounces the Ray Cones method is used.
__device__ inline float3 mipmap_sample_ray_differentials(
	const Material & material,
	int mesh_id,
	int triangle_id,
	float3 & triangle_position_edge_1,
	float3 & triangle_position_edge_2,
	const float2 & triangle_tex_coord_edge_1,
	const float2 & triangle_tex_coord_edge_2,
	const float3 & ray_direction,
	float ray_t,
	const float2 & tex_coord
) {
	// Transform Triangle edges into world space
	Matrix3x4 world = mesh_get_transform(mesh_id);
	matrix3x4_transform_direction(world, triangle_position_edge_1);
	matrix3x4_transform_direction(world, triangle_position_edge_2);
	
	// Formulae based on Chapter 20 of Ray Tracing Gems "Texture Level of Detail Strategies for Real-Time Ray Tracing"
	float one_over_k = 1.0f / dot(cross(triangle_position_edge_1, triangle_position_edge_2), ray_direction); 

	// Formula simplified because we only do ray differentials for primary rays
	// This means the differential of the ray origin is zero and
	// the differential of the ray directon is simply the x/y axis
	float3 q = ray_t * camera.x_axis;
	float3 r = ray_t * camera.y_axis;

	float3 c_u = cross(triangle_position_edge_2, ray_direction);
	float3 c_v = cross(ray_direction, triangle_position_edge_1);

	// Differentials of barycentric coordinates (u,v)
	float du_dx = one_over_k * dot(c_u, q);
	float du_dy = one_over_k * dot(c_u, r);
	float dv_dx = one_over_k * dot(c_v, q);
	float dv_dy = one_over_k * dot(c_v, r);

	// Differentials of Texture coordinates (s,t)
	float ds_dx = du_dx * triangle_tex_coord_edge_1.x + dv_dx * triangle_tex_coord_edge_2.x;
	float ds_dy = du_dy * triangle_tex_coord_edge_1.x + dv_dy * triangle_tex_coord_edge_2.x;
	float dt_dx = du_dx * triangle_tex_coord_edge_1.y + dv_dx * triangle_tex_coord_edge_2.y;
	float dt_dy = du_dy * triangle_tex_coord_edge_1.y + dv_dy * triangle_tex_coord_edge_2.y;

	float2 dx = make_float2(ds_dx, dt_dx);
	float2 dy = make_float2(ds_dy, dt_dy);

	return material.albedo(tex_coord.x, tex_coord.y, dx, dy); // Anisotropic filtering
}

// Samples the given Material's albedo texture map at the appropriate LOD
// The LOD is determined using ray cones.
__device__ inline float3 mipmap_sample_ray_cones(
	const Material & material,
	int triangle_id,
	const float3 & ray_direction,
	float ray_t,
	float cone_width,
	const float3 & hit_normal,
	const float2 & tex_coord
) {
	float lod = triangle_get_lod(triangle_id) + log2f(cone_width / fabsf(dot(ray_direction, hit_normal)));

	return material.albedo(tex_coord.x, tex_coord.y, lod); // Trilinear filtering
}
#endif
