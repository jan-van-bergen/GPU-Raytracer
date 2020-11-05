#include "BitArray.h"

#include <assert.h>

void BitArray::init(int size) {
	buffer_size = (size + 31) / 32;
	buffer = new unsigned[buffer_size];
}

void BitArray::free() {
	delete [] buffer;
}

BitArray::Access BitArray::operator[](int index) {
	int slot_index  = index / 32;
	int slot_offset = index % 32;

	assert(slot_index >= 0 && slot_index < buffer_size);

	Access access;
	access.array = this;
	access.slot_index  = slot_index;
	access.slot_offset = slot_offset;

	return access;
}
