#pragma once

struct GBuffer {
	unsigned gbuffer;
	unsigned gbuffer_position;
	unsigned gbuffer_normal;
	unsigned gbuffer_uv;
	unsigned gbuffer_triangle_id;
	unsigned gbuffer_depth;

	void init(int width, int height);

	void bind();
	void unbind();
};
