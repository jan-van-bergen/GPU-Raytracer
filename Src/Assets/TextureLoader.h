#pragma once
#include "Pathtracer/Texture.h"

namespace TextureLoader {
	bool load_dds(const char * filename, Texture & texture);
	bool load_stb(const char * filename, Texture & texture);
}
