#pragma once
#include <SDL2/SDL_scancode.h>

namespace Input {
	void update();

	void mouse_position(int * x, int * y);

	enum struct MouseButton {
		LEFT   = 1,
		MIDDLE = 2,
		RIGHT  = 3
	};

	bool is_mouse_down(MouseButton button = MouseButton::LEFT); // Is button currently down
	
	bool is_mouse_pressed (MouseButton button = MouseButton::LEFT); // Is key currently down but up last frame
	bool is_mouse_released(MouseButton button = MouseButton::LEFT); // Is key currently up but down last frame

	bool is_key_down(SDL_Scancode key); // Is key currently down
	
	bool is_key_pressed (SDL_Scancode key); // Is key currently down but up last frame
	bool is_key_released(SDL_Scancode key); // Is key currently up but down last frame
}
