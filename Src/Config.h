#pragma once
#include "CUDA/Common.h"

#include "Util/Array.h"
#include "Util/String.h"

inline Config config = { };

struct SceneConfig {
	Array<String> scene_filenames;
	String        sky_filename;
};
inline SceneConfig scene_config = { };
