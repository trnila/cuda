#include <stdio.h>
#include <stdlib.h>
#include "check.h"

void check(cudaError_t err, const char *file, int line) {
	if(err != cudaSuccess) {
		fprintf(stderr, "Error: %s in %s:%d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}