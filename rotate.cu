#include <cuda_runtime.h>
#include "check.h"

__global__ void rotate(uchar4* array, int w, int h) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= w || y >= h) {
		return;
	}

	uchar4 a = array[y * w + x];
	array[y * w + x] = array[(h - y - 1) * w + x];
	array[(h - y - 1) * w + x] = a;
}

void cuda_rotate(uchar4 *img, int w, int h) {
	uchar4 *cuda = NULL;
	checkErr(cudaMalloc(&cuda, sizeof(uchar4) * w * h));
	checkErr(cudaMemcpy(cuda, img, sizeof(uchar4)*w*h, cudaMemcpyHostToDevice));

	int block = 20;
	dim3 blocks((w+block) / block, h/(2*block));
	dim3 threads(block, block);

	rotate<<<blocks, threads>>>(cuda, w, h);
	checkErr(cudaPeekAtLastError());
	checkErr(cudaMemcpy(img, cuda, sizeof(uchar4)*w*h, cudaMemcpyDeviceToHost));
	checkErr(cudaFree(cuda));
}


