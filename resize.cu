#include <cuda_runtime.h>
#include "check.h"


#include <stdio.h>
__global__ void kernel(uchar4 *orig, uchar4 *resized, int w, int h) {
	int x = blockDim.x * blockIdx.x;
	int y = blockDim.y * blockIdx.y;

	uchar4 a = {0, 0, 0, 0};
	int vx = 0;
	int vy = 0;
	int vz = 0;

	for(int i = x; i < x + 100; i++) {
		for(int j = y; j < y + 100; j++) {
			vx += orig[j * w + i].x;
			vy += orig[j * w + i].y;
			vz += orig[j * w + i].z;

			//a.x = (a.x + orig[j * w + i].x) % 255;
			//a.y = (a.y + orig[j * w + i].y) % 255;
			//a.z = (a.z + orig[j * w + i].z) % 255;
		}
	}

	//a.x = b / 100;
	//a.x = (vx / 100) % 255;
	//a.y = vy / 100;
	//a.z = vz / 100;
	a.x = 255;

	printf("%d\n", a.x);

	resized[y * w + x] = a;
}

void cuda_resize(uchar4 *img, uchar4 *dst, int w, int h, int w1, int h1) {
	uchar4 *orig = NULL;
	uchar4 *resized = NULL;
	checkErr(cudaMalloc(&orig, sizeof(uchar4) * w * h));
	checkErr(cudaMalloc(&resized, sizeof(uchar4) * w1 * h1));

	int count = 10;
	dim3 blocks(w / count, h / count);
	dim3 threads(count, count);
	kernel<<<blocks, threads>>>(orig, resized, w1, h1);
	checkErr(cudaPeekAtLastError());
	checkErr(cudaMemcpy(dst, resized, sizeof(uchar4)*w1*h1, cudaMemcpyDeviceToHost));
	checkErr(cudaFree(orig));
	checkErr(cudaFree(resized));

}