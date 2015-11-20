#ifndef CHECK_H
	#define CHECK_H
	#include <cuda_runtime.h>

	#define checkErr(result) check(result, __FILE__, __LINE__)
	void check(cudaError_t err, const char *file, int line);
#endif