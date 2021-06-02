#version 450

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in int  in_triangle_id;

layout (location = 0) out      vec3 out_normal;
layout (location = 1) out      vec2 out_uv;
layout (location = 2) out flat int  out_triangle_id;
layout (location = 3) out      vec4 out_screen_position;
layout (location = 4) out      vec4 out_screen_position_prev;

uniform vec2 jitter;

uniform mat4 view_projection;
uniform mat4 view_projection_prev;

uniform mat4 transform;
uniform mat4 transform_prev;

void main() {
	out_normal = (transform * vec4(in_normal, 0.0f)).xyz;

	int vertex_index = gl_VertexID % 3;
	if (vertex_index == 0) {
		out_uv = vec2(0.0f, 0.0f);
	} else if (vertex_index == 1) {
		out_uv = vec2(1.0f, 0.0f);
	} else {
		out_uv = vec2(0.0f, 1.0f);
	}

	out_triangle_id = in_triangle_id;

	out_screen_position      = view_projection      * transform      * vec4(in_position, 1.0f);
	out_screen_position_prev = view_projection_prev * transform_prev * vec4(in_position, 1.0f);

	gl_Position = out_screen_position + vec4(jitter * out_screen_position.w, 0.0f, 0.0f);
}
