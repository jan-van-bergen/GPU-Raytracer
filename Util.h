#pragma once

#define INVALID -1

#define DATA_PATH(file_name) "./Data/" file_name

#define DEG_TO_RAD(angle) ((angle) * PI / 180.0f)
#define RAD_TO_DEG(angle) ((angle) / PI * 180.0f)

#define KILO_BYTE(value) (value) * 1024
#define MEGA_BYTE(value) (value) * 1024 * 1024
#define GIGA_BYTE(value) (value) * 1024 * 1024 * 1024

#define FORCEINLINE __forceinline

#define ALLIGNED_MALLOC(size, align) _aligned_malloc(size, align)
#define ALLIGNED_FREE(ptr)           _aligned_free(ptr)

namespace Util {
	const char * get_path(const char * file_path);
	
	bool file_exists(const char * filename);

	// Checks if file_check is newer than file_reference
	bool file_is_newer(const char * file_reference, const char * file_check);

	template<typename T>
	void swap(T & a, T & b) {
		T temp = a;
		a = b;
		b = temp;
	}

	void export_ppm(const char * file_path, int width, int height, const unsigned char * data);

}