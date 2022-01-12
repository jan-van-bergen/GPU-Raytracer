#pragma once
#include "CUDA/Common.h"

#include "Core/Array.h"
#include "Core/String.h"

inline Config config = { };

struct SceneConfig {
	Array<String> scene_filenames;
	String        sky_filename;
};
inline SceneConfig scene_config = { };
