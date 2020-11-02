#include "Texture.h"

#include <unordered_map>
#include <ctype.h>

#include <mutex>
#include <thread>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image/stb_image.h>

#include "Math.h"
#include "Vector4.h"

#include "CUDA_Source\Common.h"

/*
	Mipmap filter code based on http://number-none.com/product/Mipmapping,%20Part%201/index.html and https://github.com/castano/nvidia-texture-tools
*/

struct FilterBox {
	static constexpr float width = 0.5f;

	static float eval(float x) {
		if (fabsf(x) <= width) {
			return 1.0f;
		} else {
			return 0.0f;
		}
	}
};

struct FilterLanczos {
	static constexpr float width = 3.0f;

	static float eval(float x) {
		if (fabsf(x) < width) {
			return Math::sincf(PI * x) * Math::sincf(PI * x / width);
		} else {
			return 0.0f;
		}
	}
};

struct FilterKaiser {
	static constexpr float width   = 7.0f;
	static constexpr float alpha   = 4.0f;
	static constexpr float stretch = 1.0f;

	static float eval(float x) {
		float t  = x / width;
		float t2 = t * t;

		if (t2 < 1.0f) {
			return Math::sincf(PI * x * stretch) * Math::bessel_0(alpha * sqrtf(1.0f - t2)) / Math::bessel_0(alpha);
		} else {
			return 0.0f;
		}
	}
};

#if MIPMAP_DOWNSAMPLE_FILTER == MIPMAP_DOWNSAMPLE_FILTER_BOX
typedef FilterBox Filter;
#elif MIPMAP_DOWNSAMPLE_FILTER == MIPMAP_DOWNSAMPLE_FILTER_LANCZOS
typedef FilterLanczos Filter;
#elif MIPMAP_DOWNSAMPLE_FILTER == MIPMAP_DOWNSAMPLE_FILTER_KAISER
typedef FilterKaiser Filter;
#endif

static float filter_sample_box(float x, float scale) {
	constexpr int   SAMPLE_COUNT     = 32;
	constexpr float SAMPLE_COUNT_INV = 1.0f / float(SAMPLE_COUNT);

	float sample = 0.5f;
	float sum    = 0.0f;

	for (int i = 0; i < SAMPLE_COUNT; i++, sample += 1.0f) {
		float p = (x + sample * SAMPLE_COUNT_INV) * scale;

		sum += Filter::eval(p);
	}

	return sum * SAMPLE_COUNT_INV;
}

static void downsample(int width_src, int height_src, int width_dst, int height_dst, const Vector4 texture_src[], Vector4 texture_dst[], Vector4 temp[]) {
	float scale_x = float(width_dst)  / float(width_src);
	float scale_y = float(height_dst) / float(height_src);

	assert(scale_x < 1.0f && scale_y < 1.0f);

	float inv_scale_x = 1.0f / scale_x;
	float inv_scale_y = 1.0f / scale_y;

	float filter_width_x = Filter::width * inv_scale_x;
	float filter_width_y = Filter::width * inv_scale_y;

	int window_size_x = int(ceilf(filter_width_x * 2.0f)) + 1;
	int window_size_y = int(ceilf(filter_width_y * 2.0f)) + 1;

	float * kernels = new float[window_size_x + window_size_y];
	float * kernel_x = kernels;
	float * kernel_y = kernels + window_size_x;

	memset(kernel_x, 0, window_size_x * sizeof(float));
	memset(kernel_y, 0, window_size_y * sizeof(float));

	float sum_x = 0.0f;
	float sum_y = 0.0f;

	// Fill horizontal kernel
	for (int x = 0; x < window_size_x; x++) {
		float sample = filter_sample_box(x - window_size_x / 2, scale_x);

		kernel_x[x] = sample;
		sum_x += sample;
	}

	// Fill vertical kernel
	for (int y = 0; y < window_size_y; y++) {
		float sample = filter_sample_box(y - window_size_y / 2, scale_y);

		kernel_y[y] = sample;
		sum_y += sample;
	}

	// Normalize kernels
	for (int x = 0; x < window_size_x; x++) kernel_x[x] /= sum_x;
	for (int y = 0; y < window_size_y; y++) kernel_y[y] /= sum_y;

	// Apply horizontal kernel
	for (int y = 0; y < height_src; y++) {
		for (int x = 0; x < width_dst; x++) {
			float center = (float(x) + 0.5f) * inv_scale_x;

			int left = int(floorf(center - filter_width_x));

			Vector4 sum = Vector4(0.0f);

			for (int i = 0; i < window_size_x; i++) {
				int index = Math::clamp(left + i, 0, width_src - 1) + y * width_src;

				sum += kernel_x[i] * texture_src[index];
			}

			temp[x * height_src + y] = sum;
		}
	}

	// Apply vertical kernel
	for (int x = 0; x < width_dst; x++) {
		for (int y = 0; y < height_dst; y++) {
			float center = (float(y) + 0.5f) * inv_scale_y;

			int top = int(floorf(center - filter_width_y));

			Vector4 sum = Vector4(0.0f);

			for (int i = 0; i < window_size_y; i++) {
				int index = x * height_src + Math::clamp(top + i, 0, height_src - 1);

				sum += kernel_y[i] * temp[index];
			}

			texture_dst[x + y * width_dst] = sum;
		}
	}

	delete [] kernels;
}

static bool load_dds(Texture & texture, const char * file_path) {
	FILE * file; fopen_s(&file, file_path, "rb");

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

	if (data == nullptr || texture.width == 0 || texture.height == 0) {
		return false;
	}

	texture.channels = 4;
	
#if ENABLE_MIPMAPPING
	int pixel_count = 0;

	int w = texture.width;
	int h = texture.height;

	while (true) {
		pixel_count += w * h;

		if (w == 1 && h == 1) break;

		if (w > 1) w /= 2;
		if (h > 1) h /= 2;
	}
#else
	int pixel_count = texture.width * texture.height;
#endif

	Vector4 * data_rgba = new Vector4[pixel_count];

	// Copy the data over into Mipmap level 0, and convert it to linear colour space
	for (int i = 0; i < texture.width * texture.height; i++) {
		data_rgba[i] = Vector4(
			Math::gamma_to_linear(float(data[i * 4    ]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 1]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 2]) / 255.0f),
			Math::gamma_to_linear(float(data[i * 4 + 3]) / 255.0f)
		);
	}

	stbi_image_free(data);

#if ENABLE_MIPMAPPING
	texture.mip_levels = 1 + int(log2f(Math::max(texture.width, texture.height)));

	int * mip_offsets = new int[texture.mip_levels];
	mip_offsets[0] = 0;

	int offset      = texture.width * texture.height;
	int offset_prev = 0;

	int level_width  = texture.width  / 2;
	int level_height = texture.height / 2;

	int level = 1;

	Vector4 * temp = new Vector4[(texture.width / 2) * texture.height]; // Intermediate storage used when performing seperable filtering

	while (true) {
#if MIPMAP_DOWNSAMPLE_FILTER == MIPMAP_DOWNSAMPLE_FILTER_BOX
		// Box filter can downsample the previous Mip level
		downsample(level_width * 2, level_height * 2, level_width, level_height, data_rgba + offset_prev, data_rgba + offset, temp);
#else
		// Other filters downsample the original Texture for better quality
		downsample(texture.width, texture.height, level_width, level_height, data_rgba, data_rgba + offset, temp);
#endif

		mip_offsets[level++] = offset * sizeof(Vector4);

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
	texture.mip_offsets = new int(0);
#endif

	texture.data = reinterpret_cast<const unsigned char *>(data_rgba);
	
	return true;
}

static std::unordered_map<std::string, int> cache;

static std::mutex       textures_mutex; // Protects Texture::textures
static std::atomic<int> textures_finished;

static void load_texture(std::string filename, int texture_id) {
	const char * file_path = filename.c_str();

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

	Texture texture;
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
		texture.data = reinterpret_cast<const unsigned char *>(new Vector4(1.0f, 0.0f, 1.0f, 1.0f));
		texture.format = Texture::Format::RGBA;
		texture.width  = 1;
		texture.height = 1;
		texture.channels = 4;
		texture.mip_levels  = 1;
		texture.mip_offsets = new int(0);
	}
	
	{
		std::lock_guard<std::mutex> lock(textures_mutex);

		Texture::textures[texture_id] = texture;
	}

	textures_finished++;

	delete [] file_extension;
}

int Texture::load(const char * file_path) {
	int & texture_id = cache[file_path];

	// If the cache already contains this Texture simply return its index
	if (texture_id != 0) return texture_id - 1;
	
	// Otherwise, create new Texture and load it from disk
	texture_id = textures.size() + 1;

	{
		std::lock_guard<std::mutex> lock(textures_mutex);

		textures.emplace_back();
	}

	std::thread loader(load_texture, std::string(file_path), texture_id - 1);
	loader.detach();

	return texture_id - 1;
}

void Texture::wait_until_textures_loaded() {
	using namespace std::chrono_literals;

	while (textures_finished < textures.size()) {
		std::this_thread::sleep_for(100ms);
	}
}

void Texture::free() {
	delete [] data;
	delete [] mip_offsets;
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

int Texture::get_width_in_bytes(int mip_level) const {
	int level_width = Math::max(width >> mip_level, 1);

	if (format == Format::RGBA) {
		return level_width * sizeof(Vector4);
	} else {
		return level_width * channels * 4;
	}
}
