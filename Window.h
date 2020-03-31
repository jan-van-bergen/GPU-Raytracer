#pragma once
#include <cstdio>

#include <cuda.h>

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Common.h"

#include "Shader.h"

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

	void begin_gui() const;

	void update();
};
