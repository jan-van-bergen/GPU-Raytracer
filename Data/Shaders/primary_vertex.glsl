#version 420

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_uv;
layout (location = 3) in int  in_triangle_id;

layout (location = 0) out      vec3 out_normal;
layout (location = 1) out      vec2 out_uv;
layout (location = 2) out flat int  out_triangle_id;
layout (location = 3) out      vec4 out_screen_position;
layout (location = 4) out      vec4 out_screen_position_prev;

uniform mat4 view_projection;
uniform mat4 view_projection_prev;

void main() {
	out_normal      = in_normal;
	out_uv          = in_uv;
	out_triangle_id = in_triangle_id;

	out_screen_position      = view_projection      * vec4(in_position, 1.0f);
	out_screen_position_prev = view_projection_prev * vec4(in_position, 1.0f);

	gl_Position = view_projection * vec4(in_position, 1.0f);
}
