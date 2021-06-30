#pragma once

#define DATA_PATH(file_name) "./Data/" file_name

#define DEG_TO_RAD(angle) ((angle) * PI / 180.0f)
#define RAD_TO_DEG(angle) ((angle) / PI * 180.0f)

#define KILO_BYTE(value) (value) * 1024
#define MEGA_BYTE(value) (value) * 1024 * 1024
#define GIGA_BYTE(value) (value) * 1024 * 1024 * 1024

#define FORCEINLINE __forceinline

#define ALLIGNED_MALLOC(size, align) _aligned_malloc(size, align)
#define ALLIGNED_FREE(ptr)           _aligned_free(ptr)

#define MALLOCA(type, count) reinterpret_cast<type *>(_malloca(count * sizeof(type)))
#define FREEA(ptr) _freea(ptr)

namespace Util {
	void get_path(const char * filename, char * path);
	
	bool file_exists(const char * filename);

	// Checks if file_check is newer than file_reference
	bool file_is_newer(const char * file_reference, const char * file_check);

	char * file_read(const char * filename);

	const char * file_get_extension(const char * filename);

	template<typename T>
	void swap(T & a, T & b) {
		T temp = a;
		a = b;
		b = temp;
	}

	template<typename T, int N>
	constexpr int array_element_count(const T (& array)[N]) {
		return N;
	}

	void export_ppm(const char * file_path, int width, int height, const unsigned char * data);
}
