#pragma once
#include "Array.h"
#include "Compare.h"

template<typename T, typename Cmp = Compare::LessThan<T>>
struct MinHeap {
	Array<T> data;
	Cmp cmp;

	constexpr MinHeap(Cmp cmp = { }) : cmp(cmp) { }

	constexpr void insert(T item) {
		size_t size = data.size();
		data.push_back(std::move(item));
		heapify_up(size);
	}

	template<typename ... Args>
	constexpr void emplace(Args && ... args) {
		size_t size = data.size();
		data.emplace_back(std::forward<Args>(args) ...);
		heapify_up(size);
	}

	constexpr T pop() {
		ASSERT(size() > 0);
		T result = data[0];

		if (size() >= 2) {
			data[0] = data[size() - 1];
			data.pop_back();
			heapify_down(0);
		} else {
			data.pop_back();
		}

		return result;
	}

	constexpr const T & peek() const {
		ASSERT(size() > 0);
		return data[0];
	}

	constexpr size_t size() const { return data.size(); }

private:
	constexpr void heapify_up(size_t index) {
		if (index == 0) return;

		size_t parent = (index - 1) / 2;

		if (!cmp(data[parent], data[index])) {
			swap(index, parent);
			heapify_up(parent);
		}
	}

	constexpr void heapify_down(size_t index) {
		size_t left  = 2 * index + 1;
		size_t right = 2 * index + 2;

		size_t smallest = index;

		if (left  < size() && cmp(data[left],  data[smallest])) smallest = left;
		if (right < size() && cmp(data[right], data[smallest])) smallest = right;

		if (smallest != index) {
			swap(index, smallest);
			heapify_down(smallest);
		}
	}

	constexpr void swap(size_t index_a, size_t index_b) {
		T tmp = data[index_a];
		data[index_a] = data[index_b];
		data[index_b] = tmp;
	}
};
