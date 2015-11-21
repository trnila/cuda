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

	//cuda_rotate((uchar3*) img.data, w, h);
	//cuda_grayscale((uchar3*) img.data, w, h);


	int rw = w/2;
	int rh = h/2;
	uchar3 *resized = cuda_resize((uchar3*) img.data, w, h, rw, rh);

	Mat smaller(rh, rw, CV_8UC3);
	smaller.data = (uchar*) resized;

	imshow("Image1", img);
	imshow("Image", smaller);
	waitKey(0);
}
