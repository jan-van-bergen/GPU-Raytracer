#pragma once
#include <string.h>

// Hash Map using linear probing
template<typename Key, typename Value, typename Hash = std::hash<Key>, typename Cmp = std::equal_to<Key>>
struct HashMap {
	struct Map {
		size_t count    = 0;
		size_t capacity = 0;

		size_t * hashes = nullptr;
		char   * keys   = nullptr;
		char   * values = nullptr;

		constexpr void init(size_t cap) {
			constexpr size_t MIN_CAPACITY = 4;

			count    = 0;
			capacity = cap < MIN_CAPACITY ? MIN_CAPACITY : cap;

			hashes = new size_t[capacity];
			keys   = new char  [capacity * sizeof(Key)];
			values = new char  [capacity * sizeof(Value)];

			memset(hashes, 0, capacity * sizeof(size_t));
		}

		constexpr void free() {
			Key   * ks = get_keys();
			Value * vs = get_values();

			for (size_t i = 0; i < capacity; i++) {
				if (hashes[i]) {
					ks[i].~Key();
					vs[i].~Value();
				}
			}

			delete [] hashes;
			delete [] keys;
			delete [] values;
		}

		Key   * get_keys  () const { return reinterpret_cast<Key   *>(keys); }
		Value * get_values() const { return reinterpret_cast<Value *>(values); }
	} map;

	constexpr HashMap(size_t cap = 0) {
		map.init(cap);
	}

	~HashMap() {
		map.free();
	}

	constexpr Value & insert(const Key & key, const Value & value) {
		return insert(Hash()(key), key, value);
	}

	constexpr Value & insert(size_t hash, const Key & key, const Value & value) {
		if (2 * map.count >= map.capacity) {
			grow(2 * map.capacity);
		}
		return insert(map, hash, key, value);
	}

	constexpr bool try_get(const Key & key, Value & value) const {
		if (map.count == 0) return false;

		Value * value_ptr = get(map, Hash()(key), key);

		if (value_ptr) {
			value = *value_ptr;
			return true;
		} else {
			return false;
		}
	}

	constexpr Value & operator[](const Key & key) {
		size_t  hash  = Hash()(key);
		Value * value = get(map, hash, key);

		if (value) return *value;

		return insert(hash, key, Value { });
	}

	constexpr void clear() {
		map.free();
		map.init(0);
	}

	struct Iterator {
		Map  * map   = nullptr;
		size_t index = 0;

		Key   & get_key()   const { return map->get_keys  ()[index]; }
		Value & get_value() const { return map->get_values()[index]; }

//		Value & operator* () { return  map->values[index]; }
//		Value * operator->() { return &map->values[index]; }

		void operator++() {
			if (map) {
				while (index < map->capacity) {
					index++;

					if (map->hashes[index]) break;
				}
			}
			map   = nullptr;
			index = 0;
		}
		void operator--() {
			if (map) {
				while (index > 0) {
					index--;

					if (map->hashes[index]) break;
				}
			}
			map   = nullptr;
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
	static constexpr Value * get(const Map & map, size_t hash, const Key & key) {
		Cmp cmp = { };
		size_t i = hash;

		while (true) {
			i &= map.capacity - 1;

			if (map.hashes[i] == hash && cmp(map.get_keys()[i], key)) {
				return &map.get_values()[i];
			} else if (map.hashes[i] == 0) {
				return nullptr;
			}

			i++;
		}
	}

	constexpr Value & insert(Map & map, size_t hash, const Key & key, const Value & value) {
		Cmp cmp = { };
		size_t i = hash;

		while (true) {
			i &= map.capacity - 1;

			if (map.hashes[i] == 0) {
				map.hashes[i] = hash;
				new (&map.get_keys()  [i]) Key(key);
				new (&map.get_values()[i]) Value(value);
				map.count++;
				return map.get_values()[i];
			} else if (map.hashes[i] == hash && cmp(map.get_keys()[i], key)) {
				// Replace existing
				return map.get_values()[i] = value;
			}

			i++;
		}
	}

	constexpr void grow(size_t new_capacity) {
		Map new_map = { };
		new_map.init(new_capacity);

		for (size_t i = 0; i < map.capacity; i++) {
			if (map.hashes[i]) {
				insert(new_map, map.hashes[i], map.get_keys()[i], map.get_values()[i]);
			}
		}

		map.free();
		map = new_map;
	}
};
