#pragma once
#include <assert.h>
#include <string.h>

#include <new>
#include <utility>
#include <initializer_list>

template<typename T>
struct Array {
	using value_type      = T;
	using reference       = T &;
	using const_reference = const T &;
	using size_type       = size_t;

	char * buffer = nullptr;
	
	size_t count    = 0;
	size_t capacity = 0;

	constexpr Array(int initial_count = 0) {
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
	
	constexpr Array(Array && array) {
		buffer   = array.buffer;
		count    = array.count;
		capacity = array.capacity;

		array.buffer   = nullptr;
		array.count    = 0;
		array.capacity = 0;
	}

	constexpr Array & operator=(Array && array) {
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
		if (count > 0) {
			data()[--count].~T();;
		}
	}

	constexpr void resize(size_t new_count) {
		char * new_buffer = new char[new_count * sizeof(T)];

		if (buffer) {
			size_t num_move = std::min(count, new_count);
			
			// Move these elements over to the new buffer
			for (size_t i = 0; i < num_move; i++) {
				new (new_buffer + i * sizeof(T)) T(std::move(data()[i]));
			}
			// Destroy these elements from the old buffer
			for (size_t i = num_move; i < count; i++) {
				data()[i].~T();
			}

			delete [] buffer;
		}		
		buffer = new_buffer;

		// Construct new elements (if any)
		for (size_t i = count; i < new_count; i++) {
			data()[i] = { };
		}

		count    = new_count;
		capacity = new_count;
	}
	
	constexpr void reserve(size_t new_capacity) {
		if (new_capacity <= capacity) return;

		char * new_buffer = new char[new_capacity * sizeof(T)];

		if (buffer) {
			for (size_t i = 0; i < count; i++) {
				new (new_buffer + i * sizeof(T)) T(std::move(data()[i]));
			}
			delete [] buffer;
		}
		buffer = new_buffer;

		capacity = new_capacity;
	}

	constexpr void clear() {
		for (size_t i = 0; i < count; i++) {
			data()[i].~T();
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

	constexpr T & operator[](int index) {
		assert(0 <= index && index < count);
		return data()[index];
	}
	
	constexpr const T & operator[](int index) const {
		assert(0 <= index && index < count);
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
			for (size_t i = 0; i < count; i++) {
				data()[i].~T();
			}

			delete [] buffer;
			buffer = nullptr;
		}
	}

	constexpr void grow_if_needed() {
		if (count == capacity) {
			reserve(capacity + capacity / 2 + 1);
		}
	}
};
