#pragma once
#include "Pathtracer/Texture.h"

#include "Util/StringView.h"

namespace TextureLoader {
	bool load_dds(const String & filename, Texture & texture);
	bool load_stb(const String & filename, Texture & texture);
}
