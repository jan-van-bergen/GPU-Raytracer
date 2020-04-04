#version 450

layout (location = 0) in vec2 in_uv;

layout (location = 0) out vec3 out_colour;

uniform sampler2D screen;

vec3 tonemap_reinhard(vec3 colour) {
	return colour / (1.0f + colour);
}

// Based on: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
vec3 tonemap_aces(vec3 colour) {
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;

    return clamp((colour * (a * colour + b)) / (colour * (c * colour + d) + e), 0.0f, 1.0f);
}

void main() {
	vec3 colour = texture2D(screen, in_uv).rgb;

	// Tone mapping
	// colour = tonemap_aces(colour);

	// Gamma correction
	// colour = pow(colour, vec3(1.0f / 2.2f));

	out_colour = colour;
}
