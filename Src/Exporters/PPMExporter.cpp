#include "PPMExporter.h"

#include "Core/IO.h"

#include "Math/Math.h"

static Array<unsigned char> convert(int pitch, int width, int height, const Array<Vector3> & data) {
	Array<unsigned char> result = { };
	result.reserve(width * height * 3);

	// NOTE: Image needs to be flipped vertically
	for (int y = height - 1; y >= 0; y--) {
		for (int x = 0; x < width; x++) {
			result.push_back(unsigned char(Math::clamp(data[x + y * pitch].x * 255.0f, 0.0f, 255.0f)));
			result.push_back(unsigned char(Math::clamp(data[x + y * pitch].y * 255.0f, 0.0f, 255.0f)));
			result.push_back(unsigned char(Math::clamp(data[x + y * pitch].z * 255.0f, 0.0f, 255.0f)));
		}
	}

	return result;
}

void PPMExporter::save(const String & filename, int pitch, int width, int height, const Array<Vector3> & data) {
	Array<unsigned char> bytes = convert(pitch, width, height, data);

	// Based on: https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file
	FILE * file = nullptr;
	fopen_s(&file, filename.data(), "wb");

	if (!file) {
		IO::print("PPMExporter: Failed to export '{}'!\n"_sv, filename);
		return;
	}

	fprintf(file, "P6\n %d\n %d\n %d\n", width, height, 255);
	fwrite(bytes.data(), sizeof(unsigned char), width * height * 3, file);

	fclose(file);
}
