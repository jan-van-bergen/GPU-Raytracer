#pragma once
#include "Camera.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"

struct Pathtracer {
protected:
	Camera camera;
	int frames_since_camera_moved = -1;

	CUDAModule module;

	void init(const char * cuda_src_name, const char * scene_name, const char * sky_name);

public:
	void update(float delta, const unsigned char * keys);
};
