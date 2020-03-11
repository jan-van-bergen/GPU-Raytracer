#include "GBuffer.h"

#include <cassert>

#include <GL/glew.h>

void GBuffer::init(int width, int height) {
	// Create the FBO
	glGenFramebuffers(1, &gbuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gbuffer);

	// Colour Texture
	glGenTextures(1, &buffer_position);
	glGenTextures(1, &buffer_normal);
	glGenTextures(1, &buffer_uv);
	glGenTextures(1, &buffer_triangle_id);
	glGenTextures(1, &buffer_motion);

	// Initialize Position Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_position);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buffer_position, NULL);

	// Initialize Normal Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_normal);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, buffer_normal, NULL);
	
	// Initialize UV Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_uv);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, buffer_uv, NULL);

	// Initialize Triangle ID Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_triangle_id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0, GL_RED_INTEGER, GL_INT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, buffer_triangle_id, NULL);
	
	// Initialize UV Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_motion);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, buffer_motion, NULL);

	// Depth / Stencil Buffer
	glGenTextures(1, &buffer_depth);
	glBindTexture(GL_TEXTURE_2D, buffer_depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, buffer_depth, NULL);

	// Attach Draw Buffers
	GLenum draw_buffers[] = {
		GL_COLOR_ATTACHMENT0, // Position
		GL_COLOR_ATTACHMENT1, // Normal
		GL_COLOR_ATTACHMENT2, // UV
		GL_COLOR_ATTACHMENT3, // Triangle ID
		GL_COLOR_ATTACHMENT4  // Motion
	};
	glDrawBuffers(5, draw_buffers);

	// Assert correctness of the FrameBuffer
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void GBuffer::bind() {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gbuffer);
}

void GBuffer::unbind() {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}