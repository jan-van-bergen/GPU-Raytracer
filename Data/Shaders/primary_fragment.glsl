#version 420

layout (location = 0) in      vec3 in_position;
layout (location = 1) in      vec3 in_normal;
layout (location = 2) in      vec2 in_uv;
layout (location = 3) in flat int  in_triangle_id;

layout (location = 0) out vec4 out_position;
layout (location = 1) out vec4 out_normal;
layout (location = 2) out vec2 out_uv;
layout (location = 3) out int  out_triangle_id;

void main() {
	out_position    = vec4(in_position, 0.0f);
	out_normal      = vec4(in_normal,   0.0f);
	out_uv          = in_uv;
	out_triangle_id = in_triangle_id + 1; // Add one so 0 means no hit
}
