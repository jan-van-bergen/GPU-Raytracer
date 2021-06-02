#include "Window.h"

#include <cstdio>

#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_sdl.h>
#include <Imgui/imgui_impl_opengl3.h>

#include "Util/Util.h"
#include "../CUDA_Source/Common.h"

static void GLAPIENTRY gl_message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar * message, const void * user_param) {
	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n", type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "", type, severity, message);
}

Window::Window(const char * title) {
	SDL_Init(SDL_INIT_EVERYTHING);

	SDL_GL_SetAttribute(SDL_GL_RED_SIZE,     8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,   8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,    8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE,   8);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	window  = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_OPENGL);
	context = SDL_GL_CreateContext(window);
	
	SDL_SetWindowResizable(window, SDL_TRUE);

	SDL_GL_SetSwapInterval(0);

	GLenum status = glewInit();
	if (status != GLEW_OK) {
		printf("Glew failed to initialize!\n");
		abort();
	}
	
	puts("OpenGL Info:");
	printf("Version:  %s\n", glGetString(GL_VERSION));
	printf("GLSL:     %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
	printf("Vendor:   %s\n", glGetString(GL_VENDOR));
	printf("Renderer: %s\n", glGetString(GL_RENDERER));
	puts("");

#if false
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(gl_message_callback, nullptr);
#endif

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);

	width  = SCREEN_WIDTH;
	height = SCREEN_HEIGHT;
	
	glGenTextures(1, &frame_buffer_handle);

	glBindTexture(GL_TEXTURE_2D, frame_buffer_handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

	shader = Shader::load(DATA_PATH("Shaders/screen_vertex.glsl"), DATA_PATH("Shaders/screen_fragment.glsl"));
	shader.bind();
	
	glUniform1i(shader.get_uniform("screen"), 0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();

	// Setup Platform/Renderer bindings
	ImGui_ImplSDL2_InitForOpenGL(window, context);
	ImGui_ImplOpenGL3_Init("#version 450");

	ImGui::StyleColorsDark();
}

Window::~Window() {
	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void Window::render_framebuffer() const {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	shader.bind();

	glBindTexture(GL_TEXTURE_2D, frame_buffer_handle);

	// Draws a single Triangle, without any buffers
	// The Vertex Shader makes sure positions + uvs work out
	glDrawArrays(GL_TRIANGLES, 0, 3);

	shader.unbind();
}

void Window::gui_begin() const {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplSDL2_NewFrame(window);
	ImGui::NewFrame();
}

void Window::gui_end() const {
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Window::swap() {
	SDL_GL_SwapWindow(window);

	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		ImGui_ImplSDL2_ProcessEvent(&event);

		switch (event.type) {
			case SDL_WINDOWEVENT: {
				if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
					width  = event.window.data1;
					height = event.window.data2;
					
					glDeleteTextures(1, &frame_buffer_handle);
					glGenTextures   (1, &frame_buffer_handle);

					glBindTexture(GL_TEXTURE_2D, frame_buffer_handle);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
					glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

					glViewport(0, 0, width, height);

					if (resize_handler) resize_handler(frame_buffer_handle, width, height);
				}
				
				break;
			}

			case SDL_QUIT: is_closed = true; break;
		}
	}
}

void Window::read_frame_buffer(unsigned char * data) const {
	glMemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT);
	glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, data);
}
