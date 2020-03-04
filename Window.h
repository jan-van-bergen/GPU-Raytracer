#pragma once
#include <cstdio>

#include <cuda.h>

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Common.h"

#include "Shader.h"

// OpenGL Debug Callback function to report errors
inline void GLAPIENTRY glMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar * message, const void * userParam) {
	printf("GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n", type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "", type, severity, message);

	__debugbreak();
}

struct Window {
private:
	SDL_Window *  window;
	SDL_GLContext context;
	
	Shader shader;

public:
	GLuint frame_buffer_handle;

	bool is_closed = false;

	Window(const char * title);
	~Window();

	void clear() const;

	void update(bool rasterize);
};
