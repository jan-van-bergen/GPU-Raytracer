#include "Window.h"

#include <cstring>

#include <cudaGL.h>

#include "CUDACall.h"

Window::Window(int width, int height, const char * title) : 
	width(width), height(height), 
	tile_count_x(width  / tile_width), 
	tile_count_y(height / tile_height)
{
	SDL_Init(SDL_INIT_EVERYTHING);

	SDL_GL_SetAttribute(SDL_GL_RED_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_ALPHA_SIZE, 8);
	SDL_GL_SetAttribute(SDL_GL_BUFFER_SIZE, 32);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	window  = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL);
	context = SDL_GL_CreateContext(window);

	SDL_GL_SetSwapInterval(0);

	GLenum status = glewInit();
	if (status != GLEW_OK) {
		printf("Glew failed to initialize!\n");
		abort();
	}

	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);

	glEnable(GL_FRAMEBUFFER_SRGB);

	glViewport(0, 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 4.0f);

	float * data = new float[4 * width * height];
	glGenTextures(1, &frame_buffer_handle);

	glBindTexture(GL_TEXTURE_2D, frame_buffer_handle);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, data);
	glBindTexture(GL_TEXTURE_2D, 0);

	// Setup camera
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	// Init CUDA stuff
	CUDACALL(cuInit(0));

	int device_count;
	CUDACALL(cuDeviceGetCount(&device_count));

	unsigned gl_device_count;
	CUdevice * devices = new CUdevice[device_count];

	CUDACALL(cuGLGetDevices(&gl_device_count, devices, device_count, CU_GL_DEVICE_LIST_ALL));
	
	CUdevice device = devices[0];

	CUcontext context;
	CUDACALL(cuGLCtxCreate(&context, 0, device));

	delete [] devices;
	
	int major, minor;
	CUDACALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
	CUDACALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
	cuda_compute_capability = major * 10 + minor;

	CUgraphicsResource cuda_frame_buffer_handle; 
	CUDACALL(cuGraphicsGLRegisterImage(&cuda_frame_buffer_handle, frame_buffer_handle, GL_TEXTURE_2D, CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST));
	CUDACALL(cuGraphicsMapResources(1, &cuda_frame_buffer_handle, 0));

	CUDACALL(cuGraphicsSubResourceGetMappedArray(&cuda_frame_buffer, cuda_frame_buffer_handle, 0, 0));
                
	CUDACALL(cuGraphicsUnmapResources(1, &cuda_frame_buffer_handle, 0));
}

Window::~Window() {
	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void Window::clear() {
	//memset(frame_buffer, 0, width * height * sizeof(unsigned));
}

void Window::update() {
	glClear(GL_COLOR_BUFFER_BIT);
	
	glBindTexture(GL_TEXTURE_2D, frame_buffer_handle);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, frame_buffer);
	
	// Draw screen filling quad
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
	glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
	glEnd();

	SDL_GL_SwapWindow(window);

	SDL_Event e;
	while (SDL_PollEvent(&e)) {
		if (e.type == SDL_QUIT) {
			is_closed = true;
		}
	}
}
