#pragma once
#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Shader.h"

typedef void (*ResizeHandler)(unsigned frame_buffer_handle, int width, int height);

struct Window {
private:
	SDL_Window *  window;
	SDL_GLContext context;
	
	Shader shader;

public:
	GLuint frame_buffer_handle;

	int width;
	int height;

	bool is_closed = false;

	Window(const char * title);
	~Window();

	void render_framebuffer() const;

	void gui_begin() const;
	void gui_end()   const;

	void swap();

	void read_frame_buffer(unsigned char * data) const;

	ResizeHandler resize_handler = nullptr;
};
