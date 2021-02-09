#pragma once

struct BitArray {
private:
	unsigned * buffer;
	int        buffer_size;

public:
	void init(int size); // Size in bits
	void free();

	void set_all(bool value);

	struct Access {
	private:
		BitArray * array;
		int slot_index;
		int slot_offset;

		friend BitArray;

	public:
		operator bool() {
			return (array->buffer[slot_index] >> slot_offset) & 1;
		}

		void operator=(bool value) {
			unsigned & slot = array->buffer[slot_index];
			slot &= ~(1 << slot_offset);
			slot |= (value << slot_offset);
		}
	};

	Access operator[](int index); // Index in bits
};
