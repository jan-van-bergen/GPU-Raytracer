#pragma once
#include <cstdio>

#include <cuda.h>

#include <GL/glew.h>
#include <SDL2/SDL.h>

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

	void draw_quad() const;

	void gui_begin() const;
	void gui_end()   const;

	void swap();

	void read_frame_buffer(unsigned char * data) const;
};
