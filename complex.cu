#include <cuda_runtime.h>
#include "check.h"

__global__ void fkernel(uchar3 *orig, uchar3 *resized, int w, int h) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	if(x >= w || y >= h) {
		return;
	}


	int sx = w / 2;
	int sy = h / 2;

	float theta = 30 * 3.14 / 180;
	int x2 = (x-sx) * cos(theta) - (y - sy) * sin(theta) + sx;
	int y2 = (x-sx) * sin(theta) + (y - sy) * cos(theta) + sy;

	if(x2 >= 0 && x2 < w && y2 >=0 && y2 < h) {
		resized[y * w + x] = orig[y2 * w + x2];
	} else {
		resized[y * w + x] = (uchar3) {0, 0, 0};
	}
}

uchar3* cuda_slope(uchar3 *img, int w, int h) {
	uchar3 *dst = new uchar3[w * h];

	uchar3 *orig = NULL;
	uchar3 *resized = NULL;
	checkErr(cudaMalloc(&orig, sizeof(uchar3) * w * h));
	checkErr(cudaMalloc(&resized, sizeof(uchar3) * w * h));

	checkErr(cudaMemcpy(orig, img, sizeof(uchar3) * w * h, cudaMemcpyHostToDevice));

	int count = 10;
	dim3 blocks((w + count)/ count, (h + count) / count);
	dim3 threads(count, count);
	fkernel<<<blocks, threads>>>(orig, resized, w, h);
	checkErr(cudaPeekAtLastError());
	checkErr(cudaMemcpy(dst, resized, sizeof(uchar3)*w*h, cudaMemcpyDeviceToHost));
	checkErr(cudaFree(orig));
	checkErr(cudaFree(resized));

	return dst;
}