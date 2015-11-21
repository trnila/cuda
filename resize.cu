#include <cuda_runtime.h>
#include "check.h"

__global__ void kernel(uchar3 *orig, uchar3 *resized, int w, int h, int w1, int h1) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= w1 || y >= h1) {
		return;
	}

	uchar3 px1 = orig[y*2 * w + x*2];
	uchar3 px2 = orig[(y*2 + 1) * w + x*2 + 1];

	resized[y * w1 + x].x = (px1.x + px2.x) / 2;
	resized[y * w1 + x].y = (px1.y + px2.y) / 2;
	resized[y * w1 + x].z = (px1.z + px2.z) / 2;
}

uchar3* cuda_resize(uchar3 *img, int w, int h, int w1, int h1) {
	uchar3 *dst = new uchar3[w1 * h1];

	uchar3 *orig = NULL;
	uchar3 *resized = NULL;
	checkErr(cudaMalloc(&orig, sizeof(uchar3) * w * h));
	checkErr(cudaMalloc(&resized, sizeof(uchar3) * w1 * h1));

	checkErr(cudaMemcpy(orig, img, sizeof(uchar3) * w * h, cudaMemcpyHostToDevice));

	int count = 10;
	dim3 blocks((w1 + count)/ count, (h1 + count) / count);
	dim3 threads(count, count);
	kernel<<<blocks, threads>>>(orig, resized, w, h, w1, h1);
	checkErr(cudaPeekAtLastError());
	checkErr(cudaMemcpy(dst, resized, sizeof(uchar3)*w1*h1, cudaMemcpyDeviceToHost));
	checkErr(cudaFree(orig));
	checkErr(cudaFree(resized));

	return dst;
}