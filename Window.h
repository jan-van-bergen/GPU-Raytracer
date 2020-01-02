#pragma once
#include <cstdio>

#include <cuda.h>

#include <GL/glew.h>
#include <SDL2/SDL.h>

// OpenGL Debug Callback function to report errors
inline void GLAPIENTRY glMessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar * message, const void * userParam) {
	printf("GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n", type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "", type, severity, message);

	__debugbreak();
}

#define SCREEN_WIDTH  800
#define SCREEN_HEIGHT 480

struct Window {
private:
	SDL_Window *  window;
	SDL_GLContext context;
	
	GLuint frame_buffer_handle;

public:
	const int width;
	const int height;
	
	const int tile_width  = 32;
	const int tile_height = 32;

	const int tile_count_x;
	const int tile_count_y;

	bool is_closed = false;

	CUarray  cuda_frame_buffer;
	unsigned cuda_compute_capability;

	Window(int width, int height, const char * title);
	~Window();

	void clear();

	void update();

	inline void set_title(const char * title) {
		SDL_SetWindowTitle(window, title);
	}
};
