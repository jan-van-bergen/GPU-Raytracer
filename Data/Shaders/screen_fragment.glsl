#version 420

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec3 out_colour;

uniform sampler2D screen;

// Get screen, gamma corrected
vec3 get_screen(vec2 uv) {
	return pow(texture2D(screen, uv).rgb, vec3(1.0f / 2.2f));
}

void main() {
	out_colour = get_screen(in_uv);
}
