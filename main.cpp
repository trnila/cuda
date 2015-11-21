#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "cuda.h"
using namespace cv;

int main() {
	Mat img = imread("img.jpg", CV_LOAD_IMAGE_COLOR);
	if(! img.data ) {
		std::cout << "err" << std::endl;
		return 1;
	}

	int w = img.size().width;
	int h = img.size().height;

	cuda_rotate((uchar3*) img.data, w, h);
	cuda_grayscale((uchar3*) img.data, w, h);

	imshow("Image", img);
	waitKey(0);
}
