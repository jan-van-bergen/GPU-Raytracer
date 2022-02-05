#pragma once
#include <string.h>

#include <new>
#include <utility>
#include <initializer_list>

#include "Assertion.h"

template<typename T>
struct Array {
	using value_type      = T;
	using reference       = T &;
	using const_reference = const T &;
	using size_type       = size_t;

	char * buffer = nullptr;

	size_t count    = 0;
	size_t capacity = 0;

	constexpr Array(size_t initial_count = 0) {
		resize(initial_count);
	}

	constexpr Array(std::initializer_list<T> && list) {
		resize(list.size());

		size_t i = 0;
		for (T const & elem : list) {
			new (&data()[i++]) T(std::move(elem));
		}
	}

	~Array() {
		destroy_buffer();
	}

	constexpr Array(const Array & array) {
		resize(array.count);
		for (size_t i = 0; i < count; i++) {
			data()[i] = array.data()[i];
		}
	}

	constexpr Array & operator=(const Array & array) {
		resize(array.count);
		for (size_t i = 0; i < count; i++) {
			data()[i] = array.data()[i];
		}
		return *this;
	}

	constexpr Array(Array && array) noexcept {
		buffer   = array.buffer;
		count    = array.count;
		capacity = array.capacity;

		array.buffer   = nullptr;
		array.count    = 0;
		array.capacity = 0;
	}

	constexpr Array & operator=(Array && array) noexcept {
		destroy_buffer();

		buffer   = array.buffer;
		count    = array.count;
		capacity = array.capacity;

		array.buffer   = nullptr;
		array.count    = 0;
		array.capacity = 0;

		return *this;
	}

	constexpr T & push_back(T element) {
		grow_if_needed();
		return *(new (&data()[count++]) T(std::move(element)));
	}

	template<typename ... Args>
	constexpr T & emplace_back(Args && ... args) {
		grow_if_needed();
		return *(new (&data()[count++]) T { std::forward<Args>(args) ... });
	}

	constexpr void pop_back() {
		ASSERT(count > 0);
		data()[--count].~T();
	}

	constexpr void resize(size_t new_count) {
		if (new_count == count) return;

		char * new_buffer = new char[new_count * sizeof(T)];

		if (buffer) {
			size_t num_move = std::min(count, new_count);

			if constexpr (std::is_trivially_copyable_v<T>) {
				memcpy(new_buffer, buffer, num_move * sizeof(T));
			} else {
				// Move these elements over to the new buffer
				for (size_t i = 0; i < num_move; i++) {
					new (new_buffer + i * sizeof(T)) T(std::move(data()[i]));
					data()[i].~T();
				}
				// Destroy these elements from the old buffer
				for (size_t i = num_move; i < count; i++) {
					data()[i].~T();
				}
			}

			delete [] buffer;
		}
		buffer = new_buffer;

		if constexpr (std::is_constructible_v<T>) {
			// Construct new elements (if any)
			for (size_t i = count; i < new_count; i++) {
				new(data() + i) T();
			}
		}

		count    = new_count;
		capacity = new_count;
	}

	constexpr void resize_if_smaller(size_t new_count) {
		if (count < new_count) {
			resize(new_count);
		}
	}

	constexpr void reserve(size_t new_capacity) {
		if (new_capacity <= capacity) return;

		char * new_buffer = new char[new_capacity * sizeof(T)];

		if (buffer) {
			if constexpr (std::is_trivially_copyable_v<T>) {
				memcpy(new_buffer, buffer, count * sizeof(T));
			} else {
				for (size_t i = 0; i < count; i++) {
					new (new_buffer + i * sizeof(T)) T(std::move(data()[i]));
					data()[i].~T();
				}
			}
			delete [] buffer;
		}
		buffer = new_buffer;

		capacity = new_capacity;
	}

	constexpr void clear() {
		if constexpr (!std::is_trivially_destructible_v<T>) {
			// Fire destructors in reverse order
			for (size_t i = count - 1; i < count; i--) {
				data()[i].~T();
			}
		}
		count = 0;
	}

	constexpr size_t size() const { return count; }

	constexpr T * data() {
		return reinterpret_cast<T *>(buffer);
	}

	constexpr const T * data() const {
		return reinterpret_cast<const T *>(buffer);
	}

	constexpr T & operator[](size_t index) {
		ASSERT(0 <= index && index < count);
		return data()[index];
	}

	constexpr const T & operator[](size_t index) const {
		ASSERT(0 <= index && index < count);
		return data()[index];
	}

	constexpr       T * begin()       { return data(); }
	constexpr const T * begin() const { return data(); }
	constexpr       T * end  ()       { return data() + count; }
	constexpr const T * end  () const { return data() + count; }

	constexpr       T & front()       { return data()[0]; }
	constexpr const T & front() const { return data()[0]; }
	constexpr       T & back ()       { return data()[count - 1]; }
	constexpr const T & back () const { return data()[count - 1]; }

private:
	constexpr void destroy_buffer() {
		if (buffer) {
			if (!std::is_trivially_destructible_v<T>) {
				for (size_t i = count - 1; i < count; i--) {
					data()[i].~T();
				}
			}

			delete [] buffer;
			buffer = nullptr;
		}
	}

	constexpr void grow_if_needed() {
		if (count == capacity) {
			reserve(capacity + capacity / 2 + 16);
		}
	}
};
