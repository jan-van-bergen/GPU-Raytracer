#include "Input.h"

#include <cstring>

#include <SDL2/SDL_keyboard.h>

#define KEY_TABLE_SIZE SDL_NUM_SCANCODES

static bool keyboard_state_previous_frame[KEY_TABLE_SIZE] = { };

void Input::update() {
	// Save current Keyboard State
	memcpy(keyboard_state_previous_frame, SDL_GetKeyboardState(nullptr), KEY_TABLE_SIZE);
}

bool Input::is_key_down(SDL_Scancode key) {
	return SDL_GetKeyboardState(nullptr)[key];
}

bool Input::is_key_up(SDL_Scancode key) {
	return !SDL_GetKeyboardState(nullptr)[key];
}

bool Input::is_key_pressed(SDL_Scancode key) {
	return SDL_GetKeyboardState(nullptr)[key] && !keyboard_state_previous_frame[key];
}

bool Input::is_key_released(SDL_Scancode key) {
	return !SDL_GetKeyboardState(nullptr)[key] && keyboard_state_previous_frame[key];
}
