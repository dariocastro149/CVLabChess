#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//ID depending on system, default is 0
#define WEBCAM_ID 1

const string WINDOW = "Setting up the Board";
int thresholdValue1 = 50;
int thresholdValue2 = 145;
int slider1 = 50;
int slider2 = 145;

//Functions
void recognizeBoard(Mat img);
Mat cannyBoard(Mat img);
Mat contourBoard(Mat img);
Mat houghBoard(Mat img);

static void on_trackbar1(int, void*)
{
	thresholdValue1 = slider1; //set the global variable to the current slidervalue;
}

static void on_trackbar2(int, void*)
{
	thresholdValue2 = slider2; //set the global variable to the current slidervalue;
}

int main() {

	VideoCapture cap(WEBCAM_ID);
	namedWindow(WINDOW);
	Mat img, img_scanned, img_static_scanned;

	string path = "Ressources/chessboard2.png";
	Mat img_board = imread(path);
	img_static_scanned = contourBoard(cannyBoard(img_board));
	imshow("static keypoints", img_static_scanned);

	createTrackbar("Threshold Canny Min", WINDOW, &slider1, 250, on_trackbar1);
	createTrackbar("Threshold Canny Max", WINDOW, &slider2, 250, on_trackbar2);

	while (true) {
		cap.read(img);
		img_scanned = contourBoard(cannyBoard(img));
		imshow("Image", img);
		imshow(WINDOW, img_scanned);
		waitKey(1);
	}
	
	return 0;
}

void warpBoard(Mat img) {
	//TODO
}

//Draw blobs on the board
void recognizeBoard(Mat img) {
	SimpleBlobDetector detector;
	vector<KeyPoint> keypoints;
	Mat img_with_keypoints;

	//Detect blobs
	detector.detect(img, keypoints);

	//Draw detected blobs as green circles
	drawKeypoints(img, keypoints, img_with_keypoints, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//Show blobs
	imshow("keypoints", img_with_keypoints);
	waitKey(0);
}

Mat cannyBoard(Mat img) {
	Mat gray, blur, cannyed, dilated, eroded;
	const Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	

	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(7, 7), 0, 0);
	Canny(blur, cannyed, thresholdValue1, thresholdValue2);
	dilate(cannyed, dilated, kernel);
	erode(dilated, eroded, kernel);

	return eroded;
}

Mat contourBoard(Mat img) {
	Mat contoured;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	contoured = Mat::zeros(img.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(255, 0, 0);
		drawContours(contoured, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
	}

	return contoured;
}

Mat houghBoard(Mat img) {
	vector<Vec2f> lines;

	HoughLines(img, lines, 1, CV_PI / 180, 120, 0, 0);

	Mat houghed(img.size(), CV_8UC1, Scalar(0, 0, 0));
	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(houghed, pt1, pt2, Scalar(255, 255, 255), 1);
	}

	return houghed;
}