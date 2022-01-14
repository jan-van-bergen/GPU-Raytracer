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
