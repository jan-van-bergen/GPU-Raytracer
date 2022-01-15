#pragma once
#include <functional>

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Core/Array.h"

#include "Util/Shader.h"

struct Window {
	SDL_Window *  window;
	SDL_GLContext context;

	Shader shader;

	GLuint frame_buffer_handle;

	int width;
	int height;

	bool is_closed = false;

	Window(const char * title, int width, int height);
	~Window();

	void resize(int new_width, int new_height);

	void render_framebuffer() const;

	void gui_begin() const;
	void gui_end()   const;

	void swap();

	Array<unsigned char> read_frame_buffer(int & window_pitch) const;

	std::function<void(unsigned frame_buffer_handle, int width, int height)> resize_handler;
};
