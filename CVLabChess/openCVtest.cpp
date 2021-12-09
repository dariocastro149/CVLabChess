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
const float WIDTH = 500;
const float HEIGHT = 500;

int threshMinCanny = 50;
int threshMaxCanny = 145;
int threshMinArea = 1000;
int threshMaxArea = 2000;

int sliderMinCanny = 50;
int sliderMaxCanny = 145;
int sliderMinArea = 1000;
int sliderMaxArea = 2000;

//Functions
void recognizeBoard(Mat img);
Mat cannyBoard(Mat img);
Mat contourBoard(Mat img);
Mat houghBoard(Mat img);
Mat warpBoard(Mat img, vector<Point> points, float width, float height);
vector<Point> getMaxRect(Mat img);
void drawRect(Mat img, vector<Point> points);

static void on_trackbar_min_canny(int, void*)
{
	threshMinCanny = sliderMinCanny; //set the global variable to the current slidervalue;
}

static void on_trackbar_max_canny(int, void*)
{
	threshMaxCanny = sliderMaxCanny; //set the global variable to the current slidervalue;
}

static void on_trackbar_min_area(int, void*)
{
	threshMinCanny = sliderMinArea; //set the global variable to the current slidervalue;
}

static void on_trackbar_max_area(int, void*)
{
	threshMaxCanny = sliderMaxArea; //set the global variable to the current slidervalue;
}

int main() {

	VideoCapture cap(WEBCAM_ID);
	namedWindow(WINDOW);
	Mat img, img_scanned, img_static_resized, img_static_warped;

	string path = "Ressources/chessboard2.png";
	Mat img_board = imread(path);
	resize(img_board, img_static_resized, { 600 , 400 });
	vector<Point> maxRect = getMaxRect(img_static_resized);
	img_static_warped = warpBoard(img_static_resized, maxRect, 500, 500);

	drawRect(img_static_resized, maxRect);
	imshow("static keypoints", img_static_resized);
	imshow("static warped", img_static_warped);

	

	createTrackbar("Min Canny", WINDOW, &sliderMinCanny, 250, on_trackbar_min_canny);
	createTrackbar("Max Canny", WINDOW, &sliderMaxCanny, 250, on_trackbar_max_canny);
	createTrackbar("Min Area", WINDOW, &sliderMinArea, 2500, on_trackbar_min_area);
	createTrackbar("Max Area", WINDOW, &sliderMaxArea, 2500, on_trackbar_max_area);

	while (true) {
		cap.read(img);
		img_scanned = contourBoard(cannyBoard(img));
		imshow("Image", img);
		imshow(WINDOW, img_scanned);
		waitKey(1);
	}
	
	return 0;
}

vector<Point> getMaxRect(Mat img) {
	Mat cannyed;

	cannyed = cannyBoard(img);

	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	
	findContours(cannyed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> contourPoly(contours.size());
	vector<Rect> boundaryRect(contours.size());
	Point originPoint(0, 0);

	int maxArea = 0;
	vector<Point> maxRect;

	for (int i = 0; i < contours.size(); i++) {
		int area = contourArea(contours[i]);

		if (area > 500) {
			float perimeter = arcLength(contours[i], true);
			approxPolyDP(contours[i], contourPoly[i], 0.02 * perimeter, true);
			if (area > maxArea && contourPoly[i].size() == 4) {
				maxRect = contourPoly[i];
				maxArea = area;
			}
			
		}
	}

	//reorder result points
	vector<Point> result;
	vector<int> coordinateSums, coordinateDiffs;

	for (int i = 0; i < 4; i++) {
		coordinateSums.push_back(maxRect[i].x + maxRect[i].y);
		coordinateDiffs.push_back(maxRect[i].x - maxRect[i].y);
	}

	result.push_back(maxRect[min_element(coordinateSums.begin(), coordinateSums.end()) - coordinateSums.begin()]);
	result.push_back(maxRect[max_element(coordinateDiffs.begin(), coordinateDiffs.end()) - coordinateDiffs.begin()]);
	result.push_back(maxRect[max_element(coordinateSums.begin(), coordinateSums.end()) - coordinateSums.begin()]);
	result.push_back(maxRect[min_element(coordinateDiffs.begin(), coordinateDiffs.end()) - coordinateDiffs.begin()]);

	return result;
}

void drawRect(Mat img, vector<Point> points) {
	vector<vector<Point>> maxRectArray = { points };
	drawContours(img, maxRectArray, 0, Scalar(0, 255, 0), 2);

	Scalar red = Scalar(0, 0, 255);

	for (int i = 0; i < points.size(); i++) {
		circle(img, points[i], 5, red, FILLED);
		putText(img, to_string(i), points[i], FONT_HERSHEY_PLAIN, 2, red, 2);
	}
}

Mat warpBoard(Mat img, vector<Point> points, float width, float height) {
	Mat warped;
	Point2f src[4] = {points[0], points[1], points[2], points[3]};
	Point2f dst[4] = { {0.0f, 0.0f}, {width, 0.0f}, {width, height}, {0.0f, height} };

	Mat matrix = getPerspectiveTransform(src, dst);
	warpPerspective(img, warped, matrix, Point(width, height));

	return warped;
}

//Draw blobs on the board
void recognizeBoard(const Mat img) {
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

Mat cannyBoard(const Mat img) {
	Mat gray, blur, cannyed, dilated, eroded;
	const Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	

	cvtColor(img, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(7, 7), 0, 0);
	Canny(blur, cannyed, threshMinCanny, threshMaxCanny);
	dilate(cannyed, dilated, kernel);
	erode(dilated, eroded, kernel);

	return eroded;
}

Mat contourBoard(const Mat img) {
	Mat contoured;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	contoured = Mat::zeros(img.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		int area = contourArea(contours[i]);
		cout << area << endl;
		Scalar color = Scalar(255, 0, 0);
		if (area > threshMinArea && area < threshMaxArea) {
			drawContours(contoured, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
		}
	}

	return contoured;
}

Mat houghBoard(const Mat img) {
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