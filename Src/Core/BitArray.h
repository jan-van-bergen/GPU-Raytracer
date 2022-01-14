#pragma once
#include "Array.h"

struct BitArray {
private:
	using UNDERLYING_TYPE = unsigned;
	static constexpr size_t SIZEOF_UNDERLYING_TYPE_IN_BITS = sizeof(UNDERLYING_TYPE) * 8;

	Array<UNDERLYING_TYPE> buffer;

public:
	BitArray() { }

	BitArray(size_t size_int_bits) {
		size_t buffer_size = (size_int_bits + SIZEOF_UNDERLYING_TYPE_IN_BITS - 1) / SIZEOF_UNDERLYING_TYPE_IN_BITS;
		buffer = Array<UNDERLYING_TYPE>(buffer_size);
	}

	BitArray(const BitArray & other) {
		buffer = other.buffer;
	}

	BitArray(BitArray && other) noexcept {
		buffer = std::move(other.buffer);
	}

	~BitArray() { }

	BitArray & operator=(const BitArray & other) {
		buffer = other.buffer;
		return *this;
	}

	BitArray & operator=(BitArray && other) noexcept {
		buffer = std::move(other.buffer);
		return *this;
	}

	void set_all(bool value) {
		memset(buffer.data(), value ? 0xff : 0x00, buffer.size() * sizeof(UNDERLYING_TYPE));
	}

	struct Access {
	private:
		BitArray * array;
		size_t slot_index;
		size_t slot_offset;

		friend BitArray;

	public:
		operator bool() {
			return (array->buffer[slot_index] >> slot_offset) & 1;
		}

		void operator=(bool value) {
			UNDERLYING_TYPE & slot = array->buffer[slot_index];
			slot &= ~(1u << slot_offset);
			slot |= (value << slot_offset);
		}
	};

	Access operator[](size_t index_in_bits) {
		size_t slot_index  = index_in_bits / SIZEOF_UNDERLYING_TYPE_IN_BITS;
		size_t slot_offset = index_in_bits % SIZEOF_UNDERLYING_TYPE_IN_BITS;

		ASSERT(slot_index >= 0 && slot_index < buffer.size());

		Access access = { };
		access.array       = this;
		access.slot_index  = slot_index;
		access.slot_offset = slot_offset;

		return access;
	}
};
