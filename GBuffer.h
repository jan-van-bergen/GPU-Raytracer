#pragma once

struct GBuffer {
	unsigned gbuffer;
	unsigned buffer_normal_and_depth;
	unsigned buffer_uv;
	unsigned buffer_uv_gradient;
	unsigned buffer_triangle_id;
	unsigned buffer_motion;
	unsigned buffer_z_gradient;
	unsigned buffer_depth;

	void init(int width, int height);

	void bind();
	void unbind();
};
