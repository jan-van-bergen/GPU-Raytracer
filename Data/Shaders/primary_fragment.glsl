#version 420

layout (location = 0) in      vec3 in_normal;
layout (location = 1) in      vec2 in_uv;
layout (location = 2) in flat int  in_triangle_id;
layout (location = 3) in      vec4 in_screen_position;
layout (location = 4) in      vec4 in_screen_position_prev;

layout (location = 0) out vec4 out_normal_and_depth;
layout (location = 1) out vec2 out_uv;
layout (location = 2) out int  out_triangle_id;
layout (location = 3) out vec2 out_screen_position_prev;
layout (location = 4) out vec2 out_depth_gradient;

void main() {
	float linear_depth = in_screen_position.z; // gl_FragCoord.z / gl_FragCoord.w;

	out_normal_and_depth.rg = normalize(in_normal).xy;
	out_normal_and_depth.b  = linear_depth;
	out_normal_and_depth.a  = in_screen_position_prev.z;
	
	out_uv          = in_uv;
	out_triangle_id = in_triangle_id + 1; // Add one so 0 means no hit

	// Perform perspective divide
	out_screen_position_prev = in_screen_position_prev.xy / in_screen_position_prev.w;

	out_depth_gradient = vec2(dFdx(linear_depth), dFdy(linear_depth));
}
