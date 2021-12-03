#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//ID depending on system, default is 0
#define WEBCAM_ID 1

int main() {

	VideoCapture cap(WEBCAM_ID);
	Mat img;

	while (true) {
		cap.read(img);
		imshow("Image", img);
		waitKey(1);
	}
	
	return 0;
}