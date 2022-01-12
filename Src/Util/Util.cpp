#include "Util.h"

#include <string.h>

#include "Core/IO.h"

// Based on: Vose - A Linear Algorithm for Generating Random Numbers with a Given Distribution (1991)
void Util::init_alias_method(int n, double p[], ProbAlias distribution[]) {
	Array<int> large(n);
	Array<int> small(n);
	int l = 0;
	int s = 0;

	for (int j = 0; j < n; j++) {
		p[j] *= double(n);
		if (p[j] < 1.0) {
			small[s++] = j;
		} else {
			large[l++] = j;
		}
	}

	while (s != 0 && l != 0) {
		int j = small[--s];
		int k = large[--l];

		distribution[j].prob  = p[j];
		distribution[j].alias = k;

		p[k] = (p[k] + p[j]) - 1.0;

		if (p[k] < 1.0) {
			small[s++] = k;
		} else {
			large[l++] = k;
		}
	}

	while (s > 0) distribution[small[--s]] = { 1.0f, -1 };
	while (l > 0) distribution[large[--l]] = { 1.0f, -1 };
}

// Based on: https://rosettacode.org/wiki/Bitmap/Write_a_PPM_file
void Util::export_ppm(const String & filename, int width, int height, const unsigned char * data) {
	FILE * file = nullptr;
	fopen_s(&file, filename.data(), "wb");

	if (file == nullptr) {
		IO::print("Failed to export '{}'!\n"sv, filename);

		return;
	}

	fprintf(file, "P6\n %d\n %d\n %d\n", width, height, 255);
	fwrite(data, sizeof(char), width * height * 3, file);

	fclose(file);
}
