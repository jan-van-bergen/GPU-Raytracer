#include "PerfTest.h"

#include "Core/IO.h"
#include "Util/StringUtil.h"

PerfTest::PerfTest(Integrator & integrator, bool enabled, StringView scene_name) : integrator(integrator), enabled(enabled) {
	index_pov    = 0;
	index_buffer = 0;

	if (Util::strstr(scene_name, "sponza"_sv)) {
		povs = &povs_sponza;
	} else if (Util::strstr(scene_name, "San_Miguel"_sv)) {
		povs = &povs_san_miguel;
	} else if (Util::strstr(scene_name, "bistro"_sv)) {
		povs = &povs_bistro;
	} else {
		this->enabled = false;
	}
}

void PerfTest::frame_begin() {
	if (!enabled) return;

	const POV & pov = (*povs)[index_pov];

	if (index_buffer == 0) {
		integrator.scene.camera.position = pov.position;
		integrator.scene.camera.rotation = pov.rotation;
		integrator.invalidated_camera = true;
		integrator.sample_index = 0;

		IO::print("POV {}\n"_sv, index_pov);
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

			FILE * file = nullptr;
			fopen_s(&file, output_file, "wb");

			if (file == nullptr) IO::exit(1);

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
