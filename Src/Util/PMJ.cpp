#include "PMJ.h"

#include <stdio.h>
#include <random>

#include "../CUDA_Source/Common.h"

#include "Util.h"
#include "Parser.h"

void PMJ::init() {
	samples = new Vector2[PMJ_NUM_SEQUENCES * PMJ_NUM_SAMPLES_PER_SEQUENCE];

	//std::random_device rd;
	//std::mt19937 engine(rd());
	//std::uniform_real_distribution<float> dist(0, 1);

	//for (int seq = 0; seq < PMJ_NUM_SEQUENCES; seq++) {
	//	Vector2 * current_sequence = samples + seq * PMJ_NUM_SAMPLES_PER_SEQUENCE;
	//	for (int i = 0; i < PMJ_NUM_SAMPLES_PER_SEQUENCE; i++) {
	//		current_sequence[i].x = dist(engine);
	//		current_sequence[i].y = dist(engine);
	//	}
	//}

	//return;

	for (int seq = 0; seq < PMJ_NUM_SEQUENCES; seq++) {
		Vector2 * current_sequence = samples + seq * PMJ_NUM_SAMPLES_PER_SEQUENCE;

		char filename[512];
		sprintf_s(filename, DATA_PATH("PMJ/4k_samples_%02d.txt"), seq + 1);

		int          file_length = 0;
		const char * file = Util::file_read(filename, file_length);

		SourceLocation location = { };
		location.file = filename;
		location.line = 1;
		location.col  = 0;

		Parser parser;
		parser.init(file, file + file_length, location);

		for (int i = 0; i < PMJ_NUM_SAMPLES_PER_SEQUENCE; i++) {
			parser.expect('(');
			current_sequence[i].x = parser.parse_float();
			parser.expect(", ");
			current_sequence[i].y = parser.parse_float();
			parser.expect("),");
			parser.parse_newline();
		}

		delete [] file;
	}
}

void PMJ::free() {
	delete [] samples;
}
