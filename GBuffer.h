#pragma once

struct GBuffer {
	unsigned gbuffer;
	unsigned buffer_position;
	unsigned buffer_normal;
	unsigned buffer_uv;
	unsigned buffer_triangle_id;
	unsigned buffer_motion;
	unsigned buffer_depth;

	void init(int width, int height);

	void bind();
	void unbind();
};
