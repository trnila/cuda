#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

#define checkErr(result) check(result, __FILE__, __LINE__)

void check(cudaError_t err, const char *file, int line) {
	if(err != cudaSuccess) {
		fprintf(stderr, "Error: %s in %s:%d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
}

__global__ void test(uchar4* array, int w, int h) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= w || y >= h) {
		return;
	}

	uchar4 a = array[y * w + x];
	array[y * w + x] = array[(h - y - 1) * w + x];
	array[(h - y - 1) * w + x] = a;
}

void run_cuda(uchar4 *img, int w, int h) {
	uchar4 *cuda = NULL;
	checkErr(cudaMalloc(&cuda, sizeof(uchar4) * w * h));
	checkErr(cudaMemcpy(cuda, img, sizeof(uchar4)*w*h, cudaMemcpyHostToDevice));

	int block = 20;
	dim3 blocks((w+block) / block, h/(2*block));
	dim3 threads(block, block);

	test<<<blocks, threads>>>(cuda, w, h);
	checkErr(cudaPeekAtLastError());
	checkErr(cudaMemcpy(img, cuda, sizeof(uchar4)*w*h, cudaMemcpyDeviceToHost));
	checkErr(cudaFree(cuda));
}


