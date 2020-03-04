#version 420

layout (location = 0) out vec2 uv;

// Based on: https://rauwendaal.net/2014/06/14/rendering-a-screen-covering-triangle-in-opengl/
void main() {
	float x = float((gl_VertexID & 1) << 2) - 1.0f;
	float y = float((gl_VertexID & 2) << 1) - 1.0f;

	uv.x =        (x + 1.0f) * 0.5f;
	uv.y = 1.0f - (y + 1.0f) * 0.5f;
	
	gl_Position = vec4(x, y, 0.0f, 1.0f);
}
