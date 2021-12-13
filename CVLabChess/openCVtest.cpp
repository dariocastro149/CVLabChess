#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//ID depending on system, default is 0
#define WEBCAM_ID 1
#define EPSILON 0.001
const int EPSILON_LINES = 10;

const string WINDOW = "Setting up the Board";
const float WIDTH = 500;
const float HEIGHT = 500;
double IMG_RATIO = 16 / 9;

int threshMinCanny = 10;
int threshMaxCanny = 100;
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
void drawLines(Mat img, vector<vector<Point>>& horizontalLines, vector<vector<Point>>& verticalLines);
void getIntersections(const vector<vector<Point>> horizontalLines, const vector<vector<Point>> verticalLines, vector<Point>& intersections);
void drawIntersections(Mat img, const vector<Point> intersections);
void getBoardFields(vector<Point> intersections, vector<Point>& boardFields);
void getHoughLines(const Mat img, vector<vector<Point>>& horizontalLines, vector<vector<Point>>& verticalLines);
void sortLines(const vector<vector<Point>> src, vector<vector<Point>>& dst);
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
	Mat img, img_scanned, img_static_resized, img_full_static_resized, img_static_warped, img_full_static_warped, img_static_cannyed, img_static_houghed, img_static_intersected;
	vector<vector<Point>> horizontalLines, verticalLines;
	vector<Point> intersections, boardFields;

	Mat img_board = imread("Ressources/chessboard_table_lamp_empty.png");
	Mat img_full_board = imread("Ressources/chessboard_table_lamp_pawns.png");
	cout << "img width: " << img_board.cols << " img height: " << img_board.rows << endl;
	IMG_RATIO = img_board.cols / img_board.rows;
	cout << IMG_RATIO << "->" << cvRound(400 * IMG_RATIO) << endl;
	resize(img_board, img_static_resized, { 600, 400 });
	resize(img_full_board, img_full_static_resized, { 600, 400 });
	cout << "img width: " << img_static_resized.cols << " img height: " << img_static_resized.rows << endl;
	vector<Point> maxRect = getMaxRect(img_static_resized);
	img_static_warped = warpBoard(img_static_resized, maxRect, 500, 500);
	img_full_static_warped = warpBoard(img_full_static_resized, maxRect, 500, 500);
	img_static_cannyed = cannyBoard(img_static_warped);
	getHoughLines(img_static_cannyed, horizontalLines, verticalLines);
	drawLines(img_static_warped, horizontalLines, verticalLines);
	getIntersections(horizontalLines, verticalLines, intersections);
	//drawIntersections(img_static_warped, intersections);
	getBoardFields(intersections, boardFields);
	drawIntersections(img_static_warped, boardFields);
	drawIntersections(img_full_static_warped, boardFields);


	drawRect(img_static_resized, maxRect);
	imshow("static keypoints", img_static_resized);
	//imshow("static cannyed", img_static_cannyed);
	imshow("static warped", img_static_warped);
	imshow("full static warped", img_full_static_warped);
	//imshow("static houghed", img_static_houghed);


	createTrackbar("Min Canny", WINDOW, &sliderMinCanny, 250, on_trackbar_min_canny);
	createTrackbar("Max Canny", WINDOW, &sliderMaxCanny, 250, on_trackbar_max_canny);
	createTrackbar("Min Area", WINDOW, &sliderMinArea, 2500, on_trackbar_min_area);
	createTrackbar("Max Area", WINDOW, &sliderMaxArea, 2500, on_trackbar_max_area);
	//TODO make Buttons with callbacks createButton("Recalculate Board", )

	while (waitKey(30) != 'q') {
		cap.read(img);
		img_scanned = contourBoard(cannyBoard(img));
		imshow("Image", img);
		//imshow(WINDOW, img_scanned);
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

void drawLines(Mat img, vector<vector<Point>>& horizontalLines, vector<vector<Point>>& verticalLines) {

	for (int i = 0; i < horizontalLines.size(); i++) {
		line(img, horizontalLines[i][0], horizontalLines[i][1], Scalar(0, 255, 0), 1);
	}

	for (int i = 0; i < verticalLines.size(); i++) {
		line(img, verticalLines[i][0], verticalLines[i][1], Scalar(0, 255, 0), 1);
	}
}

void getIntersections(const vector<vector<Point>> horizontalLines, const vector<vector<Point>> verticalLines, vector<Point>& intersections) {
	intersections = {};
	for (int vertical = 0; vertical < verticalLines.size(); vertical++) {
		for (int horizontal = 0; horizontal < horizontalLines.size(); horizontal++) {
			int x = horizontalLines[horizontal][0].x;
			int y = verticalLines[vertical][1].y;
			intersections.push_back({ x,y });
		}
	}
}

void drawIntersections(Mat img, const vector<Point> intersections) {
	Scalar red = Scalar(0, 0, 255);

	for (int i = 0; i < intersections.size(); i++) {
		cout << "Intersection " << i << ": " << intersections[i] << endl;
		circle(img, intersections[i], 5, red, FILLED);
		putText(img, to_string(i), intersections[i], FONT_HERSHEY_PLAIN, 1, red, 1);
	}
}

void getBoardFields(vector<Point> intersections, vector<Point>& boardFields) {
	int offset = sqrt(intersections.size()) + 1;
	Point topLeftPoint;
	Point bottomRightPoint;
	boardFields = {};
	int counter = 0;
	for (int i = 0; i < intersections.size() - offset; i++) {
		if (counter < offset - 2) {
			topLeftPoint = intersections[i];
			bottomRightPoint = intersections[i + offset];
			boardFields.push_back({ topLeftPoint.x + cvRound((bottomRightPoint.x - topLeftPoint.x) / 2), topLeftPoint.y + cvRound((bottomRightPoint.y - topLeftPoint.y) / 2) });
			counter++;
		}
		else {
			counter = 0;
		}
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

void getHoughLines(const Mat img, vector<vector<Point>> &horizontalLines, vector<vector<Point>> &verticalLines) {
	vector<Vec2f> lines;
	horizontalLines = {};
	verticalLines = {};

	HoughLines(img, lines, 1, CV_PI / 180, 120, 0, 0);

	Mat houghed(img.size(), CV_8UC1, Scalar(0, 0, 0));

	for (size_t i = 0; i < lines.size(); i++) {
		float rho = lines[i][0], theta = lines[i][1];
		cout << "Rho:" << rho << endl;
		cout << "Theta:" << theta << endl;
		cout << "PI/2:" << CV_PI / 2 << endl;

		// check which houghLines have 0° or 180° angle
		if (theta < EPSILON) {
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			pt1.x = cvRound(x0 + 1000 * (-b));
			pt1.y = 0;
			pt2.x = pt1.x;
			pt2.y = WIDTH;
			cout << "Horizontal:" << pt1 << "," << pt2 << endl;
			horizontalLines.push_back({ pt1, pt2 });
		}
		else if (theta < EPSILON || abs(theta - CV_PI / 2) < EPSILON) {
			Point pt1, pt2;
			double a = cos(theta), b = sin(theta);
			double x0 = a * rho, y0 = b * rho;
			pt1.x = 0;
			pt1.y = cvRound(y0 + 1000 * (a));
			pt2.x = HEIGHT;
			pt2.y = pt1.y;
			cout << "Vertical:" << pt1 << "," << pt2 << endl;
			verticalLines.push_back({ pt1, pt2 });
		}
	}

	//sort lines
	sort(horizontalLines.begin(), horizontalLines.end(), [](const vector<Point> a, const vector<Point> b) { return a[0].x < b[0].x; });
	sort(verticalLines.begin(), verticalLines.end(), [](const vector<Point> a, const vector<Point> b) { return a[0].y < b[0].y; });

	//remove close horizontal lines
	if (horizontalLines[0][0].x < EPSILON_LINES) {
		horizontalLines.erase(horizontalLines.begin());
	}
	if ((WIDTH - horizontalLines[horizontalLines.size() - 1][0].x) < EPSILON_LINES) {
		horizontalLines.pop_back();
	}
	for (int i = 0; i < horizontalLines.size() - 1; i++) {
		if ((horizontalLines[i + 1][0].x - horizontalLines[i][0].x) < EPSILON_LINES) {
			horizontalLines.erase(horizontalLines.begin() + i);
		}
	}

	//remove close vertical lines
	if (verticalLines[0][0].y < EPSILON_LINES) {
		verticalLines.erase(verticalLines.begin());
	}
	if ((HEIGHT - verticalLines[horizontalLines.size() - 1][0].y) < EPSILON_LINES) {
		verticalLines.pop_back();
	}
	for (int i = 0; i < verticalLines.size() - 1; i++) {
		if ((verticalLines[i + 1][0].y - verticalLines[i][0].y) < EPSILON_LINES) {
			verticalLines.erase(verticalLines.begin() + i);
		}
	}
}