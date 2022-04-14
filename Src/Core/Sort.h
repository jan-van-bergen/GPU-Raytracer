#pragma once
#include "Array.h"
#include "Compare.h"

#include "Util/Util.h"

namespace Sort {
	template<typename T, typename Cmp = Compare::LessThan<T>>
	constexpr bool is_sorted(const T * first, const T * last, Cmp cmp = { }) {
		for (const T * it = first; it < last - 1; it++) {
			if (cmp(it[1], it[0])) {
				return false;
			}
		}
		return true;
	}

	template<typename T, typename Cmp = Compare::LessThan<T>>
	constexpr void quick_sort(T * first, T * last, Cmp cmp = { }) {
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
	template<typename T, typename Cmp = Compare::LessThan<T>>
	constexpr void stable_sort(T * first, T * last, T * tmp, Cmp cmp = { }) {
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

	template<typename T, typename Cmp = Compare::LessThan<T>>
	constexpr void stable_sort(T * first, T * last, Cmp cmp = { }) {
		Array<T> tmp(last - first);
		stable_sort(first, last, tmp.data(), cmp);
	}

	// Abstraction to convert any T into a Radix-sortable unsigned integer
	template<typename T>
	struct RadixSortAdapter {
		FORCE_INLINE unsigned operator()(const T & x) const {
			return unsigned(x);
		}
	};

	template<>
	struct RadixSortAdapter<int> {
		FORCE_INLINE unsigned operator()(const int & x) const {
			return unsigned(x) ^ 0x80000000;
		}
	};

	template<>
	struct RadixSortAdapter<float> {
		FORCE_INLINE unsigned operator()(const float & x) const {
			unsigned y = Util::bit_cast<unsigned>(x);
			unsigned mask = -int(y >> 31) | 0x80000000;
			return y ^ mask;
		}
	};

	template<typename T, typename Adapter = RadixSortAdapter<T>>
	void radix_sort(T * first, T * last, T * tmp, Adapter adapter = { }) {
		static constexpr unsigned NUM_RADIX_BITS = 8;
		static constexpr unsigned NUM_HISTOGRAM_BUCKETS = 31 / NUM_RADIX_BITS + 1;

		static constexpr unsigned HISTOGRAM_SIZE = 1u << NUM_RADIX_BITS;
		static constexpr unsigned HISTOGRAM_MASK = HISTOGRAM_SIZE - 1;

		size_t count = last - first;
		if (count <= 1) {
			return;
		}

		unsigned histograms[NUM_HISTOGRAM_BUCKETS][HISTOGRAM_SIZE] = { };

		// Fill histogram buckets
		for (const T * it = first; it < last; it++) {
			unsigned key = adapter(*it);
			for (unsigned b = 0; b < NUM_HISTOGRAM_BUCKETS; b++) {
				unsigned index = key >> (b * NUM_RADIX_BITS);
				histograms[b][index & HISTOGRAM_MASK]++;
			}
		}

		// Prefix sum
		unsigned sums[NUM_HISTOGRAM_BUCKETS] = { };
		for (unsigned b = 0; b < NUM_HISTOGRAM_BUCKETS; b++) {
			sums[b] = histograms[b][0];
			histograms[b][0] = 0;
		}

		unsigned sum = 0;
		for (unsigned i = 1; i < HISTOGRAM_SIZE; i++) {
			for (unsigned b = 0; b < NUM_HISTOGRAM_BUCKETS; b++) {
				sum = histograms[b][i] + sums[b];
				histograms[b][i] = sums[b];
				sums[b] = sum;
			}
		}

		// Radix passes
		T * bufs[2] = { first, tmp };

		for (unsigned b = 0; b < NUM_HISTOGRAM_BUCKETS; b++) {
			unsigned ping = b & 1;
			unsigned pong = ping ^ 1;

			if (b == NUM_HISTOGRAM_BUCKETS - 1) {
				ASSERT(pong == 0); // Last pass should output to the original buffer
			}

			T * buf_in  = bufs[ping];
			T * buf_out = bufs[pong];

			for (size_t i = 0; i < count; i++) {
				unsigned index  = adapter(buf_in[i]) >> (b * NUM_RADIX_BITS);
				unsigned offset = histograms[b][index & HISTOGRAM_MASK]++;
				buf_out[offset] = buf_in[i];
			}
		}
	}

	template<typename T, typename Adapter = RadixSortAdapter<T>>
	void radix_sort(T * first, T * last, Adapter adapter = { }) {
		Array<T> tmp = Array<T>(last - first);
		radix_sort(first, last, tmp.data(), adapter);
	}
}
