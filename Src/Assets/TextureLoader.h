#pragma once
#include "Core/StringView.h"

#include "Pathtracer/Texture.h"

namespace TextureLoader {
	bool load_dds(const String & filename, Texture & texture);
	bool load_stb(const String & filename, Texture & texture);
}
