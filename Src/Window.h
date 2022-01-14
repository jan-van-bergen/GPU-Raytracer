#pragma once
#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Core/Array.h"

#include "Util/Shader.h"

typedef void (*ResizeHandler)(unsigned frame_buffer_handle, int width, int height);

struct Window {
	SDL_Window *  window;
	SDL_GLContext context;

	Shader shader;

	GLuint frame_buffer_handle;

	int width;
	int height;

	bool is_closed = false;

	void init(const char * title, int width, int height);
	void free();

	void resize(int new_width, int new_height);

	void render_framebuffer() const;

	void gui_begin() const;
	void gui_end()   const;

	void swap();

	Array<unsigned char> read_frame_buffer(int & window_pitch) const;

	ResizeHandler resize_handler = nullptr;
};
