#ifndef CUDA_H
	#define CUDA_H

	void cuda_rotate(uchar3 *img, int w, int h);
	void cuda_grayscale(uchar3 *img, int w, int h);
	uchar3* cuda_resize(uchar3 *img, int w, int h, int w1, int h1);

#endif