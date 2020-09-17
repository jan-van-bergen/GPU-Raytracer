#include "Texture.h"

#include <unordered_map>
#include <ctype.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include "Math.h"
#include "Vector4.h"

#include "Util.h"
#include "CUDA_Source\Common.h"

/*
	Mipmap filter code based on http://number-none.com/product/Mipmapping,%20Part%201/index.html and https://github.com/castano/nvidia-texture-tools
*/

enum class FilterType {
	BOX,
	LANCZOS,
	KAISER
};

typedef float (*Filter)(float width, float x);

static float filter_box(float width, float x) {
	if (fabsf(x) <= width) {
		return 1.0f;
	} else {
		return 0.0f;
	}
}

static float filter_lanczos(float width, float x) {
	if (fabsf(x) < 3.0f) {
		return Math::sincf(PI * x) * Math::sincf(PI * x / 3.0f);
	} else {
		return 0.0f;
	}
}

static float filter_kaiser(float width, float x) {
	constexpr float alpha   = 4.0f;
	constexpr float stretch = 1.0f;

	float sinc = Math::sincf(PI * x * stretch);
	float t = x / width;
	float t2 = t * t;

	if (t2 < 1.0f) {
		return sinc * Math::bessel_0(alpha * sqrtf(1.0f - t2)) / Math::bessel_0(alpha);
	} else {
		return 0.0f;
	}
}

static float filter_sample_box(Filter filter, float filter_width, float x, float scale, int samples) {
	float sum = 0.0f;
	float inv_samples = 1.0f / float(samples);

	for (int s = 0; s < samples; s++) {
		float p = (x + (float(s) + 0.5f) * inv_samples) * scale;

		sum += filter(filter_width, p);
	}

	return sum * inv_samples;
}

static void downsample(FilterType filter_type, int width_src, int height_src, int width_dst, int height_dst, const Vector4 texture_src[], Vector4 texture_dst[], Vector4 temp[]) {
	float scale_x = float(width_dst)  / float(width_src);
	float scale_y = float(height_dst) / float(height_src);

	assert(scale_x < 1.0f && scale_y < 1.0f);

	float inv_scale_x = 1.0f / scale_x;
	float inv_scale_y = 1.0f / scale_y;

	float  filter_width_x;
	float  filter_width_y;
	Filter filter = nullptr;

	switch (filter_type) {
		case FilterType::BOX: {
			filter_width_x = 0.5f;
			filter_width_y = 0.5f;

			filter = &filter_box;

			break;
		}
		case FilterType::LANCZOS: {
			filter_width_x = 3.0f;
			filter_width_y = 3.0f;

			filter = &filter_lanczos;

			break;
		}
		case FilterType::KAISER: {
			filter_width_x = 5.0f;
			filter_width_y = 5.0f;

			filter = &filter_kaiser;

			break;
		}

		default: abort();
	}

	float width_x = filter_width_x * inv_scale_x;
	float width_y = filter_width_y * inv_scale_y;

	int window_size_x = int(ceilf(width_x * 2.0f)) + 1;
	int window_size_y = int(ceilf(width_y * 2.0f)) + 1;

	float * kernel_x = new float[window_size_x];
	float * kernel_y = new float[window_size_y];

	memset(kernel_x, 0, window_size_x * sizeof(float));
	memset(kernel_y, 0, window_size_y * sizeof(float));

	float sum_x = 0.0f;
	float sum_y = 0.0f;

	for (int j = 0; j < window_size_x; j++) {
		float sample = filter_sample_box(filter, filter_width_x, j - window_size_x / 2, scale_x, 32);

		kernel_x[j] = sample;
		sum_x += sample;
	}
	for (int j = 0; j < window_size_y; j++) {
		float sample = filter_sample_box(filter, filter_width_y, j - window_size_y / 2, scale_y, 32);

		kernel_y[j] = sample;
		sum_y += sample;
	}

	for (int j = 0; j < window_size_x; j++) kernel_x[j] /= sum_x;
	for (int j = 0; j < window_size_y; j++) kernel_y[j] /= sum_y;

	// Apply horizontal kernel
	for (int j = 0; j < height_src; j++) {
		Vector4 * row = temp + j * width_dst;

		for (int i = 0; i < width_dst; i++) {
			float center = (float(i) + 0.5f) * inv_scale_x;

			int left = int(floorf(center - width_x));

			Vector4 sum = Vector4(0.0f);

			for (int k = 0; k < window_size_x; k++) {
				int index = Math::clamp(left + k, 0, width_src - 1) + j * width_src;

				sum += kernel_x[k] * texture_src[index];
			}

			row[i] = sum;
		}
	}

	// Apply vertical kernel
	for (int i = 0; i < width_dst; i++) {
		Vector4 * col = texture_dst + i;

		for (int j = 0; j < height_dst; j++) {
			float center = (float(j) + 0.5f) * inv_scale_y;

			int top = int(floorf(center - width_y));

			Vector4 sum = Vector4(0.0f);

			for (int k = 0; k < window_size_y; k++) {
				int index = i + Math::clamp(top + k, 0, height_src - 1) * width_dst;

				sum += kernel_y[k] * temp[index];
			}

			col[j * width_dst] = sum;
		}
	}

	delete [] kernel_x;
	delete [] kernel_y;
}

static const int zero = 0;
static const Vector4 pink = Vector4(1.0f, 0, 1.0f, 1.0f);

static std::unordered_map<std::string, int> cache;

static bool load_dds(Texture & texture, const char * file_path) {
	FILE * file;
	fopen_s(&file, file_path, "rb");

	if (file == nullptr) return false;
	
	bool success = false;

	fseek(file, 0, SEEK_END);
	int file_size = ftell(file);
	fseek(file, 0, SEEK_SET);

	unsigned char header[128];
	fread_s(header, sizeof(header), 1, 128, file);

	// First four bytes should be "DDS "
	if (memcmp(header, "DDS ", 4) != 0) goto exit;

	// Get width and height
	memcpy_s(&texture.width,      sizeof(int), header + 16, sizeof(int));
	memcpy_s(&texture.height,     sizeof(int), header + 12, sizeof(int));
	memcpy_s(&texture.mip_levels, sizeof(int), header + 28, sizeof(int));

	texture.width  = (texture.width  + 3) / 4;
	texture.height = (texture.height + 3) / 4;

	if (memcmp(header + 84, "DXT", 3) != 0) goto exit;

	// Get format
	// See https://en.wikipedia.org/wiki/S3_Texture_Compression
	switch (header[87]) {
		case '1': { // DXT1
			texture.format = Texture::Format::BC1;
			texture.channels = 2;

			break;
		}
		case '3': { // DXT3
			texture.format = Texture::Format::BC2;
			texture.channels = 4;

			break;
		}
		case '5': { // DXT5
			texture.format = Texture::Format::BC3;
			texture.channels = 4;

			break;
		}

		default: goto exit; // Unsupported format
	}

	int data_size = file_size - sizeof(header);

	unsigned char * data = new unsigned char[data_size];
	fread_s(data, data_size, 1, data_size, file);

	int * mip_offsets = new int[texture.mip_levels];
	
	int block_size = texture.channels * 4;

	int level_width  = texture.width;
	int level_height = texture.height;
	int level_offset = 0;

	for (int level = 0; level < texture.mip_levels; level++) {
		if (level_width == 0 || level_height == 0) {
			texture.mip_levels = level;
			
			break;
		}

		mip_offsets[level] = level_offset;
		level_offset += level_width * level_height * block_size;

		level_width  /= 2;
		level_height /= 2;
	}
	
	texture.data = data;
	texture.mip_offsets = mip_offsets;

	success = true;

exit:
	fclose(file);

	return success;
}

static bool load_stbi(Texture & texture, const char * file_path) {
	unsigned char * data = stbi_load(file_path, &texture.width, &texture.height, &texture.channels, STBI_rgb_alpha);

	if (data == nullptr || texture.width == 0 || texture.height == 0) return false;

	texture.channels = 4;
	
	int size = texture.width * texture.height;
	Vector4 * data_rgba = new Vector4[size + size / 3];
	
#define RAINBOW false

#if RAINBOW
	const Vector4 rainbow[] = { Vector4(1,0,0,0), Vector4(1,1,0,0), Vector4(0,1,0,0), Vector4(0,1,1,0), Vector4(0,0,1,0), Vector4(1,0,1,0), Vector4(1,1,1,0) };

	for (int i = 0; i < texture.width * texture.height; i++) {
		data_rgba[i] = rainbow[0];
	}
#else
	// Copy the data over into Mipmap level 0, and convert it to linear colour space
	for (int i = 0; i < texture.width * texture.height; i++) {
		data_rgba[i] = Vector4(
			Math::gamma_to_linear(float(data[i * 4    ]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 1]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 2]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 3]) / 255.0f)
		);
	}
#endif

#if ENABLE_MIPMAPPING
	texture.mip_levels = 1 + int(log2f(Math::max(texture.width, texture.height)));

	int * mip_offsets = new int[texture.mip_levels];
	mip_offsets[0] = 0;

	int offset      = texture.width * texture.height;
	int offset_prev = 0;

	int level_width  = texture.width  / 2;
	int level_height = texture.height / 2;

	int level = 1;

	Vector4 * temp = new Vector4[texture.width * texture.height / 2];

	FilterType filter_type = FilterType::LANCZOS;

	while (true) {
#if RAINBOW
		for (int i = 0; i < level_width * level_height; i++) {
			data_rgba[offset + i] = rainbow[Math::min(Util::array_element_count(rainbow) - 1, level)];
		}
#else
		if (filter_type == FilterType::BOX) {
			// Box filter can downsample the previous Mip level
			downsample(filter_type, level_width * 2, level_height * 2, level_width, level_height, data_rgba + offset_prev, data_rgba + offset, temp);
		} else {
			// Other filters downsample the original Texture for better quality
			downsample(filter_type, texture.width, texture.height, level_width, level_height, data_rgba, data_rgba + offset, temp);
		}
#endif

		mip_offsets[level++] = offset * sizeof(Vector4);

#if false
		if (strstr(file_path, "sponza_curtain_blue_diff.tga")) {
			unsigned char * data_rgb = new unsigned char[level_width * level_height * 3];

			for (int i = 0; i < level_width * level_height; i++) {
				data_rgb[i * 3 + 0] = unsigned char(Math::clamp((data_rgba + offset)[i].x * 255.0f, 0.0f, 255.0f));
				data_rgb[i * 3 + 1] = unsigned char(Math::clamp((data_rgba + offset)[i].y * 255.0f, 0.0f, 255.0f));
				data_rgb[i * 3 + 2] = unsigned char(Math::clamp((data_rgba + offset)[i].z * 255.0f, 0.0f, 255.0f));
			}

			char path[32]; sprintf_s(path, "mip_%i.ppm", level - 1);

			Util::save_ppm(path, level_width, level_height, data_rgb);

			delete [] data_rgb;
		}
#endif

		if (level_width == 1 && level_height == 1) break;

		offset_prev = offset;
		offset += level_width * level_height;

		if (level_width  > 1) level_width  /= 2;
		if (level_height > 1) level_height /= 2;
	}

	delete [] temp;

	assert(level == texture.mip_levels);

	texture.mip_offsets = mip_offsets;
#else
	texture.mip_levels = 1;
	texture.mip_offsets = &zero;
#endif

	texture.data = reinterpret_cast<const unsigned char *>(data_rgba);
	
	return true;
}

int Texture::load(const char * file_path) {
	int & texture_id = cache[file_path];

	// If the cache already contains this Texture simply return it
	if (texture_id != 0 && textures.size() > 0) return texture_id;
	
	// Otherwise, create new Texture and load it from disk
	texture_id        = textures.size();
	Texture & texture = textures.emplace_back();

	int    file_path_length = strlen(file_path);
	char * file_extension   = nullptr;

	// Extract file extension from path
	for (int i = file_path_length - 1; i >= 0; i--) {
		if (file_path[i] == '.') {
			int file_extension_length = file_path_length - i;
			file_extension = new char[file_extension_length];

			for (int j = 0; j < file_extension_length; j++) {
				file_extension[j] = tolower(file_path[i + 1 + j]);
			}

			break;
		}
	}

	bool success = false;

	if (file_extension) {
		if (strcmp(file_extension, "dds") == 0) {
			success = load_dds(texture, file_path); // DDS is loaded using custom code
		} else {
			success = load_stbi(texture, file_path); // other file formats use stb_image
		}
	}

	if (!success) {
		printf("WARNING: Failed to load Texture '%s'!\n", file_path);

		if (texture.data) delete [] texture.data;

		// Make Texture pure pink to signify invalid Texture
		texture.data = reinterpret_cast<const unsigned char *>(&pink);
		texture.format = Texture::Format::RGBA;
		texture.width  = 1;
		texture.height = 1;
		texture.channels = 4;
		texture.mip_levels  = 1;
		texture.mip_offsets = &zero;
	}

	delete [] file_extension;

	return texture_id;
}

CUarray_format Texture::get_cuda_array_format() const {
	switch (format) {
		case Format::BC1:  return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::BC2:  return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::BC3:  return CUarray_format::CU_AD_FORMAT_UNSIGNED_INT32;
		case Format::RGBA: return CUarray_format::CU_AD_FORMAT_FLOAT;
	}
}

CUresourceViewFormat Texture::get_cuda_resource_view_format() const {
	switch (format) {
		case Texture::Format::BC1:  return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
		case Texture::Format::BC2:  return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
		case Texture::Format::BC3:  return CUresourceViewFormat::CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
		case Texture::Format::RGBA: return CUresourceViewFormat::CU_RES_VIEW_FORMAT_FLOAT_4X32;
	}
}

int Texture::get_cuda_resource_view_width() const {
	if (format == Format::RGBA) {
		return width;
	} else {
		return width * 4;
	}
}

int Texture::get_cuda_resource_view_height() const {
	if (format == Format::RGBA) {
		return height;
	} else {
		return height * 4;
	}
}

int Texture::get_width_in_bytes() const {
	if (format == Format::RGBA) {
		return width * sizeof(Vector4);
	} else {
		return width * channels * 4;
	}
}
