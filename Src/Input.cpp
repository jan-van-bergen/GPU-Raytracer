#include "Input.h"

#include <cstring>

#include <SDL2/SDL_Mouse.h>
#include <SDL2/SDL_keyboard.h>

#define KEY_TABLE_SIZE SDL_NUM_SCANCODES

static Uint32 mouse_state_prev;
static bool keyboard_state_prev[KEY_TABLE_SIZE];

void Input::update() {
	// Save current Mouse and Keyboard states
	mouse_state_prev = SDL_GetMouseState(nullptr, nullptr);
	memcpy(keyboard_state_prev, SDL_GetKeyboardState(nullptr), KEY_TABLE_SIZE);
}

void Input::mouse_position(int * x, int * y) {
	SDL_GetMouseState(x, y);
}

static bool state_get_button(Uint32 state, Input::MouseButton button) {
	return state & SDL_BUTTON((int)button);
}

bool Input::is_mouse_down(MouseButton button) {
	return state_get_button(SDL_GetMouseState(nullptr, nullptr), button);
}

bool Input::is_mouse_pressed(MouseButton button) {
	return is_mouse_down(button) && !state_get_button(mouse_state_prev, button);
}

bool Input::is_mouse_released(MouseButton button) {
	return !is_mouse_down(button) && state_get_button(mouse_state_prev, button);
}

bool Input::is_key_down(SDL_Scancode key) {
	return SDL_GetKeyboardState(nullptr)[key];
}

bool Input::is_key_pressed(SDL_Scancode key) {
	return is_key_down(key) && !keyboard_state_prev[key];
}

bool Input::is_key_released(SDL_Scancode key) {
	return !is_key_down(key) && keyboard_state_prev[key];
}
