#pragma once
#include "Array.h"
#include "Compare.h"

#include "Util/Util.h"

namespace Sort {
	template<typename T, typename Cmp>
	constexpr void quick_sort(T * first, T * last, Cmp cmp = Compare::LessThan<T>()) {
		if (first >= last - 1) return;

		T pivot = first[(last - first) / 2];

		T * i = first;
		T * j = first;
		T * k = last;

		// Dutch National Flag algorithm
		while (j < k) {
			if (cmp(*j, pivot)) { // *j < pivot
				Util::swap(*i, *j);
				i++;
				j++;
			} else if (cmp(pivot, *j)) { // *j > pivot
				k--;
				Util::swap(*j, *k);
			} else { // *j == pivot
				j++;
			}
		}

		// Recurse
		quick_sort(first, i,    cmp);
		quick_sort(j,     last, cmp);
	}

	// Merge sort
	template<typename T, typename Cmp>
	constexpr void stable_sort(T * first, T * last, T * tmp, Cmp cmp = Compare::LessThan<T>()) {
		if (last - first <= 1) return;

		T * middle = first + (last - first) / 2;
		stable_sort(first, middle, tmp, cmp);
		stable_sort(middle, last,  tmp, cmp);

		// Merge into tmp buffer
		T * head_left  = first;
		T * head_right = middle;
		size_t index = 0;

		while (head_left != middle && head_right != last) {
			if (cmp(*head_right, *head_left)) {
				tmp[index++] = std::move(*head_right++);
			} else {
				tmp[index++] = std::move(*head_left++);
			}
		}
		while (head_left  != middle) tmp[index++] = std::move(*head_left++);
		while (head_right != last)   tmp[index++] = std::move(*head_right++);

		size_t count = last - first;
		for (size_t i = 0; i < count; i++) {
			first[i] = std::move(tmp[i]);
		}
	}

	template<typename T, typename Cmp>
	constexpr void stable_sort(T * first, T * last, Cmp cmp = Compare::LessThan<T>()) {
		Array<T> tmp(last - first);
		stable_sort(first, last, tmp.data(), cmp);
	}
}
