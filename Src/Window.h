#pragma once
#include <functional>

#include <GL/glew.h>
#include <SDL2/SDL.h>

#include "Core/Array.h"
#include "Core/String.h"

#include "Util/Shader.h"
#include "Math/Math.h"

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

	void set_size(int new_width, int new_height);
	void resize_frame_buffer(int new_width, int new_height);

	void hide() { SDL_HideWindow(window); }
	void show(bool center = true) {
		SDL_ShowWindow(window);

		if (center) {
			SDL_DisplayMode display_mode = { };
			SDL_GetCurrentDisplayMode(0, &display_mode);
			int pos_x = Math::max((display_mode.w - width)  / 2, 0);
			int pos_y = Math::max((display_mode.h - height) / 2, 0);
			SDL_SetWindowPosition(window, pos_x, pos_y);
		}
	}

	void render_framebuffer() const;

	void gui_begin() const;
	void gui_end()   const;

	void swap();

	Array<Vector3> read_frame_buffer(bool hdr, int & pitch) const;

	std::function<void(unsigned frame_buffer_handle, int width, int height)> resize_handler;
};
