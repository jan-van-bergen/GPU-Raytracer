#pragma once
#include <functional>

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Core/Array.h"
#include "Core/String.h"

#include "Util/Shader.h"

struct Vector3;

struct Window {
	SDL_Window  * window;
	SDL_GLContext context;

	Shader shader;

	GLuint frame_buffer_handle;

	int width;
	int height;

	bool is_closed = false;

	Window(const String & title, int width, int height);
	~Window();

	void resize(int new_width, int new_height);

	void hide() { SDL_HideWindow(window); }
	void show() { SDL_ShowWindow(window); }

	void render_framebuffer() const;

	void gui_begin() const;
	void gui_end()   const;

	void swap();

	Array<Vector3> read_frame_buffer(bool hdr, int & pitch) const;

	std::function<void(unsigned frame_buffer_handle, int width, int height)> resize_handler;
};
