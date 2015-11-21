#include <cuda_runtime.h>
#include "check.h"

__global__ void grayscale(uchar4* array, int w, int h) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= w || y >= h) {
		return;
	}

	uchar4 px = array[y * w + x];
	px.z = px.y = px.x = px.x * 0.114 + px.y * 0.587 + px.z * 0.299;
	array[y * w + x] = px;
}

void cuda_grayscale(uchar4 *img, int w, int h) {
	uchar4 *cuda = NULL;
	checkErr(cudaMalloc(&cuda, sizeof(uchar4) * w * h));
	checkErr(cudaMemcpy(cuda, img, sizeof(uchar4)*w*h, cudaMemcpyHostToDevice));

	int block = 20;
	dim3 blocks((w+block) / block, h/(block));
	dim3 threads(block, block);

	grayscale<<<blocks, threads>>>(cuda, w, h);
	checkErr(cudaPeekAtLastError());
	checkErr(cudaMemcpy(img, cuda, sizeof(uchar4)*w*h, cudaMemcpyDeviceToHost));
	checkErr(cudaFree(cuda));
}


