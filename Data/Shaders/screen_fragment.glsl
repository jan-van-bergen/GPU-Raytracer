#version 450

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec3 out_colour;

uniform sampler2D screen;

// Get screen, gamma corrected
vec3 get_screen(vec2 uv) {
	vec3 colour = texture2D(screen, uv).rgb;

	// Tone mapping
	colour = colour / (1.0f + colour);

	// Gamma correction
	colour = pow(colour, vec3(1.0f / 2.2f));

	return colour;
}

void main() {
	out_colour = get_screen(in_uv);
}
