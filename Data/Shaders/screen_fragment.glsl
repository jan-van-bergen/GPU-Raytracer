#version 450

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec3 out_colour;

uniform sampler2D screen;

void main() {
	vec3 colour = texture2D(screen, in_uv).rgb;

	// Tone mapping
	colour = colour / (1.0f + colour);

	// Gamma correction
	colour = pow(colour, vec3(1.0f / 2.2f));

	out_colour = colour;
}
