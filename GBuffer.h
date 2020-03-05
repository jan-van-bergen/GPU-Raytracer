#pragma once

namespace GBuffer {
	inline unsigned gbuffer;
	inline unsigned gbuffer_position;
	inline unsigned gbuffer_normal;
	inline unsigned gbuffer_uv;
	inline unsigned gbuffer_triangle_id;
	inline unsigned gbuffer_depth;

	void init(int width, int height);

	void bind();
	void unbind();
}
