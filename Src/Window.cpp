#include "Window.h"

#include <cstdio>

#include <Imgui/imgui.h>
#include <Imgui/imgui_impl_sdl.h>
#include <Imgui/imgui_impl_opengl3.h>

#include "Core/IO.h"

#include "Math/Math.h"
#include "Util/Util.h"

static void GLAPIENTRY gl_message_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char * message, const void * user_param) {
	IO::print("GL CALLBACK: {} type = 0x{:x}, severity = 0x{:x}, message = {}\n"_sv, type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **"_sv : ""_sv, type, severity, message);
}

Window::Window(const String & title, int width, int height) {
	SDL_Init(SDL_INIT_EVERYTHING);

	SDL_GL_SetAttribute(SDL_GL_RED_SIZE,     8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,   8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,    8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE,   8);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	this->width  = width;
	this->height = height;

	window  = SDL_CreateWindow(title.data(), SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL | SDL_WINDOW_HIDDEN);
	context = SDL_GL_CreateContext(window);

	SDL_SetWindowResizable(window, SDL_TRUE);

	SDL_GL_SetSwapInterval(0);

	GLenum status = glewInit();
	if (status != GLEW_OK) {
		IO::print("Glew failed to initialize!\n"_sv);
		IO::exit(1);
	}

	IO::print("OpenGL Info:\n"_sv);
	IO::print("Version:  {}\n"_sv, reinterpret_cast<const char *>(glGetString(GL_VERSION)));
	IO::print("GLSL:     {}\n"_sv, reinterpret_cast<const char *>(glGetString(GL_SHADING_LANGUAGE_VERSION)));
	IO::print("Vendor:   {}\n"_sv, reinterpret_cast<const char *>(glGetString(GL_VENDOR)));
	IO::print("Renderer: {}\n"_sv, reinterpret_cast<const char *>(glGetString(GL_RENDERER)));
	IO::print('\n');

#if false
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(gl_message_callback, nullptr);
#endif

	glEnable(GL_TEXTURE_2D);
	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);

	glGenTextures(1, &frame_buffer_handle);

	glBindTexture(GL_TEXTURE_2D, frame_buffer_handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

	String shader_vertex   = IO::file_read("Src/Shaders/post.vert");
	String shader_fragment = IO::file_read("Src/Shaders/post.frag");

	shader = Shader::load(shader_vertex.view(), shader_fragment.view());
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

void Window::set_size(int new_width, int new_height) {
	SDL_SetWindowSize(window, new_width, new_height);
	resize_frame_buffer(new_width, new_height);
}

void Window::resize_frame_buffer(int new_width, int new_height) {
	width  = new_width;
	height = new_height;

	glDeleteTextures(1, &frame_buffer_handle);
	glGenTextures   (1, &frame_buffer_handle);

	glBindTexture(GL_TEXTURE_2D, frame_buffer_handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

	glViewport(0, 0, width, height);

	if (resize_handler) resize_handler(frame_buffer_handle, width, height);
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
					resize_frame_buffer(event.window.data1, event.window.data2);
				}

				break;
			}

			case SDL_QUIT: is_closed = true; break;
		}
	}
}

Array<Vector3> Window::read_frame_buffer(bool hdr, int & pitch) const {
	int pack_alignment = 0;
	glGetIntegerv(GL_PACK_ALIGNMENT, &pack_alignment);

	pitch = int(Math::round_up(width * sizeof(Vector3), size_t(pack_alignment)) / sizeof(Vector3));
	Array<Vector3> data(pitch * height);

	glMemoryBarrier(GL_PIXEL_BUFFER_BARRIER_BIT);

	if (hdr) {
		// For HDR output we use the frame_buffer Texture,
		// since this is the raw output of the Pathtracer
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, data.data());
	} else {
		// For LDR output we use the Window's actual frame buffer,
		// since this has been tonemapped and gamma corrected
		glReadPixels(0, 0, width, height, GL_RGB, GL_FLOAT, data.data());
	}

	return data;
}
