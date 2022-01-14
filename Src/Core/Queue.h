#pragma once

template<typename T>
struct Queue {
	size_t capacity = 0;
	T    * data     = nullptr;

	constexpr Queue() { }

	constexpr Queue(const Queue & other) {
		capacity = other.capacity;
		data     = new T[capacity];
		memcpy(data, other.data, capacity * sizeof(T));

		head = data + (other.head - other.data);
		tail = data + (other.tail - other.data);
	}

	constexpr Queue(Queue && other) noexcept {
		capacity = other.capacity;
		data     = other.data;
		head     = other.head;
		tail     = other.tail;

		other.capacity = 0;
		other.data     = nullptr;
		other.head     = nullptr;
		other.tail     = nullptr;
	}

	constexpr Queue & operator=(const Queue & other) {
		if (data) delete [] data;

		capacity = other.capacity;
		data     = new T[capacity];
		memcpy(data, other.data, capacity * sizeof(T));

		head = data + (other.head - other.data);
		tail = data + (other.tail - other.data);

		return *this;
	}

	constexpr Queue & operator=(Queue && other) noexcept {
		if (data) delete [] data;

		capacity = other.capacity;
		data     = other.data;
		head     = other.head;
		tail     = other.tail;

		other.capacity = 0;
		other.data     = nullptr;
		other.head     = nullptr;
		other.tail     = nullptr;

		return *this;
	}

	~Queue() {
		if (data) delete [] data;
	}

	T * head = nullptr;
	T * tail = nullptr;

	constexpr void push(T item) {
		if (!data) {
			capacity = 16;
			data = new T[capacity];

			head = data;
			tail = data;
		} else if (size() == capacity - 1) {
			size_t new_capacity = capacity + capacity / 2 + 1;
			T    * new_data = new T[new_capacity];

			T * new_head = new_data;
			T * new_tail = new_data;

			if (head <= tail) {
				for (T * it = head; it != tail; it++) {
					*new_tail++ = std::move(*it);
				}
			} else {
				for (T * it = head; it != data + capacity; it++) {
					*new_tail++ = std::move(*it);
				}
				for (T * it = data; it != tail; it++) {
					*new_tail++ = std::move(*it);
				}
			}

			delete [] data;
			data = new_data;
			capacity = new_capacity;

			head = new_head;
			tail = new_tail;
		}

		*tail = std::move(item);
		inc_wrap(tail);
	}

	constexpr T pop() {
		if (size() == 0) abort();

		T result = *head;
		inc_wrap(head);
		return result;
	}

	constexpr size_t size() const {
		if (head <= tail) {
			return tail - head;
		} else {
			return capacity - (head - tail);
		}
	}

	constexpr bool is_empty() const { return size() == 0; }

private:
	constexpr void inc_wrap(T *& ptr) {
		ptr++;
		if (ptr == data + capacity) {
			ptr = data;
		}
	}
};
