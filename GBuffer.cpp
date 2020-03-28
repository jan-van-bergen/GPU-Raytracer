#include "GBuffer.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include <GL/glew.h>

void GBuffer::init(int width, int height) {
	// Create the FBO
	glGenFramebuffers(1, &gbuffer);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gbuffer);

	// Colour Texture
	glGenTextures(1, &buffer_normal_and_depth);
	glGenTextures(1, &buffer_uv);
	glGenTextures(1, &buffer_triangle_id);
	glGenTextures(1, &buffer_motion);
	glGenTextures(1, &buffer_z_gradient);
	glGenTextures(1, &buffer_depth);

	// Initialize Normal Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_normal_and_depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, buffer_normal_and_depth, NULL);
	
	// Initialize UV Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_uv);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, buffer_uv, NULL);

	// Initialize Triangle ID Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_triangle_id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_R32I, width, height, 0, GL_RED_INTEGER, GL_INT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, buffer_triangle_id, NULL);
	
	// Initialize UV Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_motion);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, buffer_motion, NULL);

	// Initialize Z Gradient Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_z_gradient);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, NULL);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT4, GL_TEXTURE_2D, buffer_z_gradient, NULL);

	// Depth Buffer
	glBindTexture(GL_TEXTURE_2D, buffer_depth);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, buffer_depth, NULL);

	// Attach Draw Buffers
	GLenum draw_buffers[] = {
		GL_COLOR_ATTACHMENT0, // Normal in rgb, Depth in a
		GL_COLOR_ATTACHMENT1, // UV
		GL_COLOR_ATTACHMENT2, // Triangle ID
		GL_COLOR_ATTACHMENT3, // Motion
		GL_COLOR_ATTACHMENT4  // Depth Gradient  
	};
	glDrawBuffers(sizeof(draw_buffers) / sizeof(GLenum), draw_buffers);

	// Assert correctness of the FrameBuffer
	GLuint frame_buffer_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (frame_buffer_status != GL_FRAMEBUFFER_COMPLETE) {
		switch (frame_buffer_status) {
			case GL_FRAMEBUFFER_UNDEFINED:                     puts("Error while loading GBuffer: GL_FRAMEBUFFER_UNDEFINED!\n"); break;
			case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:         puts("Error while loading GBuffer: GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT!\n"); break;
			case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: puts("Error while loading GBuffer: GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT!\n"); break;
			case GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER:        puts("Error while loading GBuffer: GL_FRAMEBUFFER_INCOMPLETE_DRAW_BUFFER!\n"); break;
			case GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER:        puts("Error while loading GBuffer: GL_FRAMEBUFFER_INCOMPLETE_READ_BUFFER!\n"); break;
			case GL_FRAMEBUFFER_UNSUPPORTED:                   puts("Error while loading GBuffer: GL_FRAMEBUFFER_UNSUPPORTED!\n"); break;
			case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE:        puts("Error while loading GBuffer: GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE!\n"); break;
			case GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS:      puts("Error while loading GBuffer: GL_FRAMEBUFFER_INCOMPLETE_LAYER_TARGETS!\n"); break;
			default: break;
		}
		
		abort();
	}

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void GBuffer::bind() {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, gbuffer);
}

void GBuffer::unbind() {
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}