#pragma once
#include <SDL2/SDL_scancode.h>

namespace Input {
	void update();

	bool is_key_down(SDL_Scancode key); // Is Key currently down
	bool is_key_up  (SDL_Scancode key); // Is key currently up

	bool is_key_pressed (SDL_Scancode key); // Is Key currently down but up last frame
	bool is_key_released(SDL_Scancode key); // Is Key currently up but down last frame
}
