#version 450

layout (location = 0) in      vec3 in_normal;
layout (location = 1) in      vec2 in_uv;
layout (location = 2) in flat int  in_triangle_id;
layout (location = 3) in      vec4 in_screen_position;
layout (location = 4) in      vec4 in_screen_position_prev;

layout (location = 0) out vec4 out_normal_and_depth;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out vec4 out_uv_gradient;
layout (location = 3) out int  out_triangle_id;
layout (location = 4) out vec2 out_screen_position_prev;
layout (location = 5) out vec2 out_depth_gradient;

uniform vec2 jitter;

// Based on: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
vec2 oct_wrap(vec2 v) {
    return vec2(
		(1.0f - abs( v.y )) * ( v.x >= 0.0f ? +1.0f : -1.0f),
		(1.0f - abs( v.x )) * ( v.y >= 0.0f ? +1.0f : -1.0f)
	);
}

// Based on: https://knarkowicz.wordpress.com/2014/04/16/octahedron-normal-vector-encoding/
vec2 oct_encode_normal(vec3 n) {
	n /= ( abs( n.x ) + abs( n.y ) + abs( n.z ) );
    n.xy = n.z >= 0.0f ? n.xy : oct_wrap( n.xy );
    return n.xy * 0.5f + 0.5f;
}

void main() {
	float linear_depth = in_screen_position.z; // gl_FragCoord.z / gl_FragCoord.w;

	out_normal_and_depth.rg = oct_encode_normal(normalize(in_normal));
	out_normal_and_depth.b  = linear_depth;
	out_normal_and_depth.a  = in_screen_position_prev.z;
	
	out_uv          = in_uv;
	out_uv_gradient = vec4(dFdx(in_uv), dFdy(in_uv));

	out_triangle_id = in_triangle_id + 1; // Add one so 0 means no hit

	// Perform perspective divide and add jitter
	out_screen_position_prev = in_screen_position_prev.xy / in_screen_position_prev.w + jitter;

	out_depth_gradient = vec2(dFdx(linear_depth), dFdy(linear_depth));
}
