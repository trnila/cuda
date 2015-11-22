#include <cuda_runtime.h>
#include "check.h"

__global__ void border(uchar3* array, int w, int h) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= w || y >= h) {
		return;
	}

	if(x % 20 == 0 || x % 20 == 1 || y % 20 == 0 || y % 20 == 1) {
		array[y * w + x] = (uchar3) {255, 255, 255};
	}
}

void cuda_border(uchar3 *img, int w, int h) {
	uchar3 *cuda = NULL;
	checkErr(cudaMalloc(&cuda, sizeof(uchar3) * w * h));
	checkErr(cudaMemcpy(cuda, img, sizeof(uchar3)*w*h, cudaMemcpyHostToDevice));

	int block = 20;
	dim3 blocks((w+block) / block, h/(block));
	dim3 threads(block, block);

	border<<<blocks, threads>>>(cuda, w, h);
	checkErr(cudaPeekAtLastError());
	checkErr(cudaMemcpy(img, cuda, sizeof(uchar3)*w*h, cudaMemcpyDeviceToHost));
	checkErr(cudaFree(cuda));
}


