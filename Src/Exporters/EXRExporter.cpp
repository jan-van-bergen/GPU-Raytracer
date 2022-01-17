#include "EXRExporter.h"

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include "Core/IO.h"

#include "Util/Util.h"

void EXRExporter::save(const String & filename, int pitch, int width, int height, const Array<Vector3> & data) {
	Array<float> data_rgb[3] = {
		Array<float>(width * height),
		Array<float>(width * height),
		Array<float>(width * height)
	};

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			data_rgb[0][x + (height - 1 - y) * width] = data[x + y * pitch].x;
			data_rgb[1][x + (height - 1 - y) * width] = data[x + y * pitch].y;
			data_rgb[2][x + (height - 1 - y) * width] = data[x + y * pitch].z;
		}
	}

	const float * data_bgr[3] = {
		data_rgb[2].data(), // B
		data_rgb[1].data(), // G
		data_rgb[0].data()  // R
	};

	EXRImage image = { };
	image.num_channels = 3;
	image.images       = Util::bit_cast<unsigned char **>(&data_bgr[0]);
	image.width        = width;
	image.height       = height;

	EXRHeader header = { };
	header.num_channels = 3;

	EXRChannelInfo channels[3] = { };
	header.channels = channels;
	memcpy(header.channels[0].name, "B", 2);
	memcpy(header.channels[1].name, "G", 2);
	memcpy(header.channels[2].name, "R", 2);

	int           pixel_types[3] = { TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT, TINYEXR_PIXELTYPE_FLOAT };
	int requested_pixel_types[3] = { TINYEXR_PIXELTYPE_HALF,  TINYEXR_PIXELTYPE_HALF,  TINYEXR_PIXELTYPE_HALF};
	header.pixel_types           = pixel_types;
	header.requested_pixel_types = requested_pixel_types;

	const char * error_string = nullptr;
	int          error_code = SaveEXRImageToFile(&image, &header, filename.data(), &error_string);

	if (error_code != TINYEXR_SUCCESS) {
		IO::print("EXRExporter: Failed to export {}\n{}\n"_sv, filename, error_string);
	}

	FreeEXRErrorMessage(error_string);
}
