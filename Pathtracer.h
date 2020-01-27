#pragma once
#include "Camera.h"

#include "CUDAModule.h"
#include "CUDAKernel.h"

struct Pathtracer {
protected:
	Camera camera;
	float frames_since_camera_moved = 0.0f;

	CUDAModule module;

	void init(const char * cuda_src_name, const char * scene_name);
};
