#include "PerfTest.h"

void PerfTest::init(Pathtracer * pathtracer, bool enabled, const char * scene_name) {
	this->enabled = enabled;

	index_pov    = 0;
	index_buffer = 0;

	this->pathtracer = pathtracer;

	if (strstr(scene_name, "sponza")) {
		povs = &povs_sponza;
	} else if (strstr(scene_name, "San_Miguel")) {
		povs = &povs_san_miguel;
	} else if (strstr(scene_name, "bistro")) {
		povs = &povs_bistro;
	} else {
		this->enabled = false;
	}
}

void PerfTest::frame_begin() {
	if (!enabled) return;

	const POV & pov = (*povs)[index_pov];

	if (index_buffer == 0) {
		pathtracer->scene.camera.position = pov.position;
		pathtracer->scene.camera.rotation = pov.rotation;
		pathtracer->invalidated_camera = true;
		pathtracer->frames_accumulated = 0;
		
		printf("POV %i\n", index_pov);
	}
}

bool PerfTest::frame_end(float frame_time) {
	if (!enabled) return false;

	POV & pov = (*povs)[index_pov];

	pov.timings[index_buffer] = frame_time * 1000.0f;

	index_buffer++;
	if (index_buffer == BUFFER_SIZE) {
		index_buffer = 0;

		index_pov++;
		if (index_pov == (*povs).size()) {
			index_pov = 0;

			FILE * file; fopen_s(&file, output_file, "wb");

			if (file == nullptr) abort();

			for (int i = 0; i < (*povs).size(); i++) {
				const POV & pov = (*povs)[i];

				float sum            = 0.0f;
				float sum_of_squares = 0.0f;

				for (int j = 0; j < BUFFER_SIZE; j++) {
					sum            += pov.timings[j];
					sum_of_squares += pov.timings[j] * pov.timings[j];
				}

				float avg = sum            / float(BUFFER_SIZE);
				float var = sum_of_squares / float(BUFFER_SIZE) - avg*avg;

				fprintf_s(file, "POV %i: avg=%f, stddev=%f\n", i, avg, sqrtf(var));
			}

			fprintf_s(file, "\n");

			for (int i = 0; i < (*povs).size(); i++) {
				const POV & pov = (*povs)[i];

				fprintf_s(file, "POV %i:\n", i);

				for (int j = 0; j < BUFFER_SIZE; j++) {
					fprintf_s(file, "%f\n", pov.timings[j]);
				}
			}

			fclose(file);

			return true;
		}
	}

	return false;
}
