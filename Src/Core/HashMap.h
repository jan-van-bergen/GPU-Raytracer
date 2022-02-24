#pragma once
#include <string.h>

#include "Hash.h"
#include "Compare.h"
#include "Allocators/Allocator.h"

// Hash Map using linear probing
template<typename Key, typename Value, typename Hash = Hash<Key>, typename Cmp = Compare::Equal<Key>>
struct HashMap {
	struct Map {
		size_t count    = 0;
		size_t capacity = 0;

		size_t * hashes = nullptr;
		char   * keys   = nullptr;
		char   * values = nullptr;

		constexpr void init(Allocator * allocator, size_t cap) {
			constexpr size_t MIN_CAPACITY = 4;

			count    = 0;
			capacity = cap < MIN_CAPACITY ? MIN_CAPACITY : cap;

			hashes = Allocator::alloc_array<size_t>(allocator, capacity);
			keys   = Allocator::alloc_array<char>  (allocator, capacity * sizeof(Key));
			values = Allocator::alloc_array<char>  (allocator, capacity * sizeof(Value));

			memset(hashes, 0, capacity * sizeof(size_t));
		}

		constexpr void free(Allocator * allocator) {
			Key   * ks = get_keys();
			Value * vs = get_values();

			for (size_t i = 0; i < capacity; i++) {
				if (hashes[i]) {
					ks[i].~Key();
					vs[i].~Value();
				}
			}

			Allocator::free_array(allocator, hashes);
			Allocator::free_array(allocator, keys);
			Allocator::free_array(allocator, values);
		}

		constexpr Value & insert(size_t hash, Key key, Value value) {
			Cmp cmp = { };
			size_t i = hash;
			while (true) {
				i &= capacity - 1;
				if (hashes[i] == 0) {
					hashes[i] = hash;
					new (&get_keys()  [i]) Key(std::move(key));
					new (&get_values()[i]) Value(std::move(value));
					count++;
					return get_values()[i];
				} else if (hashes[i] == hash && cmp(get_keys()[i], key)) {
					// Replace existing
					return get_values()[i] = std::move(value);
				}
				i++;
			}
		}

		constexpr Value * get(size_t hash, const Key & key) const {
			Cmp cmp = { };
			size_t i = hash;
			while (true) {
				i &= capacity - 1;
				if (hashes[i] == hash && cmp(get_keys()[i], key)) {
					return &get_values()[i];
				} else if (hashes[i] == 0) {
					return nullptr;
				}
				i++;
			}
		}

		constexpr Key   * get_keys  () const { return reinterpret_cast<Key   *>(keys); }
		constexpr Value * get_values() const { return reinterpret_cast<Value *>(values); }
	} map;

	Allocator * allocator = nullptr;

	constexpr HashMap(Allocator * allocator = nullptr, size_t cap = 0) : allocator(allocator) {
		map.init(allocator, cap);
	}

	NON_COPYABLE(HashMap);
	NON_MOVEABLE(HashMap);

	~HashMap() {
		map.free(allocator);
	}

	constexpr Value & insert(Key key, Value value) {
		size_t hash = Hash()(key);
		return insert_by_hash(hash, std::move(key), std::move(value));
	}

	constexpr Value & insert_by_hash(size_t hash, Key key, Value value) {
		if (2 * map.count >= map.capacity) {
			grow(2 * map.capacity);
		}
		return map.insert(hash, std::move(key), std::move(value));
	}

	constexpr Value * try_get(const Key & key) const {
		return try_get_by_hash(Hash()(key), key);
	}

	constexpr Value * try_get_by_hash(size_t hash, const Key & key) const {
		return map.get(hash, key);
	}

	constexpr Value & operator[](const Key & key) {
		size_t  hash  = Hash()(key);
		Value * value = map.get(hash, key);

		if (value) return *value;

		return insert_by_hash(hash, key, Value { });
	}

	constexpr void clear() {
		map.free(allocator);
		map.init(allocator, 0);
	}

	struct Iterator {
		Map  * map   = nullptr;
		size_t index = 0;

		Key   & get_key()   const { return map->get_keys  ()[index]; }
		Value & get_value() const { return map->get_values()[index]; }

		void operator++() {
			if (map) {
				while (index + 1 < map->capacity) {
					index++;

					if (map->hashes[index]) return;
				}
				map = nullptr;
			}
			index = 0;
		}
		void operator--() {
			if (map) {
				while (index > 0) {
					index--;

					if (map->hashes[index]) return;
				}
				map = nullptr;
			}
			index = 0;
		}

		bool operator==(const Iterator & other) const { return map == other.map && index == other.index; }
		bool operator!=(const Iterator & other) const { return map != other.map || index != other.index; }
	};

	Iterator begin() {
		for (size_t i = 0; i < map.capacity; i++) {
			if (map.hashes[i]) {
				return Iterator { &map, i };
			}
		}
		return end();
	}

	Iterator end() {
		return Iterator { nullptr, 0 };
	}

private:
	constexpr void grow(size_t new_capacity) {
		Map new_map = { };
		new_map.init(allocator, new_capacity);

		for (size_t i = 0; i < map.capacity; i++) {
			if (map.hashes[i]) {
				new_map.insert(map.hashes[i], std::move(map.get_keys()[i]), std::move(map.get_values()[i]));
			}
		}

		map.free(allocator);
		map = new_map;
	}
};
