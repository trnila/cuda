#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
using namespace cv;

void cuda_rotate(uchar4 *img, int w, int h);

uchar4* matToArray(Mat &img) {
	int w = img.size().width;
	int h = img.size().height;

	uchar4 *data = new uchar4[w * h];

	for(int x = 0; x < w; x++) {
		for(int y = 0; y < h; y++) {
			Vec3b v3 = img.at<Vec3b>(y, x);
			uchar4 bgr = {  v3[ 0 ], v3[ 1 ], v3[ 2 ], 0 };

			data[y * w + x] = bgr;
		}
	}

	return data;
}

void arrayToMat(Mat &img, uchar4 *data) {
	int w = img.size().width;
	int h = img.size().height;

	for(int x = 0; x < w; x++) {
		for(int y = 0; y < h; y++) {
			int i = y * w + x;
			Vec3b v3 = {data[i].x, data[i].y, data[i].z};

			img.at<Vec3b>(y, x) = v3;
		}
	}
}

int main() {
	Mat img = imread("img.jpg", CV_LOAD_IMAGE_COLOR);
	if(! img.data ) {
		std::cout << "err" << std::endl;
		return 1;
	}

	int w = img.size().width;
	int h = img.size().height;


	uchar4* data = matToArray(img);
	cuda_rotate(data, w, h);
	arrayToMat(img, data);

	imshow("Image", img);
	waitKey(0);
}
