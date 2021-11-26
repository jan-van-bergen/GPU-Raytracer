#pragma once
#include <malloc.h>
#include <utility>

#define DATA_PATH(file_name) "./Data/" file_name

#define DEG_TO_RAD(angle) ((angle) * PI / 180.0f)
#define RAD_TO_DEG(angle) ((angle) / PI * 180.0f)

#define KILO_BYTE(value) (value) * 1024
#define MEGA_BYTE(value) (value) * 1024 * 1024
#define GIGA_BYTE(value) (value) * 1024 * 1024 * 1024

#define FORCEINLINE __forceinline

#define UNREACHABLE __assume(false)

#define ALLIGNED_MALLOC(size, align) _aligned_malloc(size, align)
#define ALLIGNED_FREE(ptr)           _aligned_free(ptr)

#define MALLOCA(type, count) reinterpret_cast<type *>(_malloca(count * sizeof(type)))
#define FREEA(ptr) _freea(ptr)

struct alignas(8) ProbAlias {
	float prob;
	int   alias;
};

namespace Util {
	void get_path(const char * filename, char * path);

	const char * get_absolute_filename(const char * path, int len_path, const char * filename, int len_filename);

	bool file_exists(const char * filename);

	bool file_is_newer(const char * filename_a, const char * filename_b);

	char * file_read(const char * filename, int & file_length);

	const char * find_last(const char * haystack, const char * needles);

	void init_alias_method(int n, double p[], ProbAlias distribution[]);

	template<typename T>
	void swap(T & a, T & b) {
		T temp = a;
		a = b;
		b = temp;
	}

	template<typename T, typename Cmp>
	constexpr void quick_sort(T * first, T * last, Cmp cmp) {
		if (first >= last - 1) return;

		T pivot = first[(last - first) / 2];

		T * i = first;
		T * j = first;
		T * k = last;

		// Dutch National Flag algorithm
		while (j < k) {
			if (cmp(*j, pivot)) { // *j < pivot
				swap(*i, *j);
				i++;
				j++;
			} else if (cmp(pivot, *j)) { // *j > pivot
				k--;
				swap(*j, *k);
			} else { // *j == pivot
				j++;
			}
		}

		// Recurse
		quick_sort(first, i,    cmp);
		quick_sort(j,     last, cmp);
	}

	template<typename T>
	constexpr void quick_sort(T * first, T * last) {
		quick_sort(first, last, [](const T & a, const T & b) { return a < b; });
	}

	// Merge sort
	template<typename T, typename Cmp>
	constexpr void stable_sort(T * first, T * last, T * tmp, Cmp cmp) {
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
	constexpr void stable_sort(T * first, T * last, Cmp cmp) {
		T * tmp = new T[last - first];
		stable_sort(first, last, tmp, cmp);
		delete [] tmp;
	}

	template<typename T>
	constexpr void stable_sort(T * first, T * last) {
		stable_sort(first, last, [](const T & a, const T & b) { return a < b; });
	}

	template<typename T, int N>
	constexpr int array_count(const T (& array)[N]) {
		return N;
	}

	void export_ppm(const char * file_path, int width, int height, const unsigned char * data);
}
