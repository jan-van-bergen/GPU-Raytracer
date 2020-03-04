#version 420

layout (location = 0) in      vec3 in_position;
layout (location = 1) in      vec3 in_normal;
layout (location = 2) in      vec2 in_uv;
layout (location = 3) in flat int  in_triangle_id;

layout (location = 0) out vec3 out_position;
// layout (location = 1) out vec3 out_normal;
// layout (location = 2) out vec2 out_uv;
// layout (location = 3) out int  out_triangle_id;

void main() {
	out_position    = 0.5 + 0.5 * in_normal;
	// out_normal      = in_normal;
	// out_uv          = in_uv;
	// out_triangle_id = in_triangle_id;
}
