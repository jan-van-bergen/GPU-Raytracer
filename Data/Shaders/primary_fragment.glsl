#version 420

layout (location = 0) in      vec3 in_position;
layout (location = 1) in      vec3 in_normal;
layout (location = 2) in      vec2 in_uv;
layout (location = 3) in flat int  in_triangle_id;
layout (location = 4) in      vec4 in_screen_position;
layout (location = 5) in      vec4 in_screen_position_prev;

layout (location = 0) out vec4  out_normal;
layout (location = 1) out vec2  out_uv;
layout (location = 2) out int   out_triangle_id;
layout (location = 3) out vec2  out_motion;
layout (location = 4) out float out_depth;
layout (location = 5) out vec2  out_depth_gradient;

uniform mat4 view_projection_prev;

void main() {
	out_normal      = vec4(normalize(in_normal), 0.0f);
	out_uv          = in_uv;
	out_triangle_id = in_triangle_id + 1; // Add one so 0 means no hit

	// @PERFORMANCE
	vec4 screen_position_prev = view_projection_prev * vec4(in_position, 1.0f);

	out_motion = screen_position_prev.xy / screen_position_prev.w;

	const float near =   0.1f;
	const float far  = 250.0f;

	out_depth = gl_FragCoord.z / gl_FragCoord.w;
	out_depth_gradient = vec2(dFdx(out_depth), dFdy(out_depth));
}
