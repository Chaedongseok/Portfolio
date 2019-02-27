#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <time.h>
#include <stdio.h>
#include <wiringPi.h>
#include <wiringShift.h>
#include <math.h>

//Set the GPIO pin number
#define PI 3.141592
#define Direction_pan ??
#define Direction_panl ??
#define Power_pan ??
#define Direction_tilt ??
#define Direction_tiltd ??
#define Power_tilt ??


using namespace std;
using namespace cv;


CascadeClassifier faceDetector1("????");//opencv 파일내에 첨부되어있는 cascade.xml파일을 연결시켜주는곳이다.
CascadeClassifier faceDetector2("????");

void Setup()
{
	pinMode(Direction_pan, OUTPUT);
	pinMode(Direction_panl,OUTPUT);
	pinMode(Power_pan, OUTPUT);	
	pinMode(Power_tilt, OUTPUT);
	pinMode(Direction_tilt, OUTPUT);
	pinMode(Direction_tiltd,OUTPUT);
	digitalWrite(Direction_pan , 0);
	digitalWrite(Direction_panl , 0);//panning을 left와 right방향으로 분류하여 변수를 저장하였다.
	digitalWrite(Power_pan, 0);
	digitalWrite(Direction_tilt , 0);//tilting을 up와 down방향으로 분류하여 변수를 저장하였다.
	digitalWrite(Direction_tiltd , 0);
	digitalWrite(Power_tilt, 0);
}

int main(int argc, char** argv)
{	
	if (wiringPiSetup()==-1)
	return 1;
	int Scale_lim = 40; //object가 중심픽셀로부터 scale_lim픽셀만큼 벗어나면 인식한다.
	int Level_p ;//panning방향
	int Level_t ;//tilt방향
	
	//int check = 0;
	int cameraNumber = 0;
	if (argc > 1)
		cameraNumber = atoi(argv[1]);
	VideoCapture camera;
	camera.open(cameraNumber);
	if (!camera.isOpened())
	{
		cerr << "ERROR: Could not access the camera or video!" << endl;
		exit(1);
	}
	namedWindow("webcam");
	int Frame_WIDTH = 640;
	int Frame_HEIGHT = 480;
	camera.set(CV_CAP_PROP_FRAME_WIDTH, Frame_WIDTH);
	camera.set(CV_CAP_PROP_FRAME_HEIGHT, Frame_HEIGHT);
	int Frame_M_WIDTH = (Frame_WIDTH / 2);
//	int Frame_M_WIDTH = 80;
	int Frame_M_HEIGHT = (Frame_HEIGHT / 2);
	int midFaceX = Frame_M_WIDTH;	//중심점을 나타낸다.
	int midFaceY = Frame_M_HEIGHT;
	
	
	char Vertical = 0;
	
	
	int Frame_num =0;
	int stepSize = 1;
	while (true)
	{
		Setup();
		Mat cameraFrame;
		camera >> cameraFrame;
		if (cameraFrame.empty())
		{
			cerr << "ERROR: Couldn't grab a camera frame." << endl;
			exit(1);
		}
		
		//convert RGB to gray
		Mat gray;
		if (cameraFrame.channels() == 3)
		{
			cvtColor(cameraFrame, gray, CV_BGR2GRAY);
		}
		else if (cameraFrame.channels() == 4)
		{
			cvtColor(cameraFrame, gray, CV_BGRA2GRAY);
		}
		else
		{
			gray = cameraFrame;
		}

		//reduction image
		Mat smallImg;
		const int DETECTION_WIDTH = 160;
		float scale = cameraFrame.cols / (float)DETECTION_WIDTH;
		if (cameraFrame.cols > DETECTION_WIDTH)
		{
			int scaleHeight = cvRound(gray.rows / scale);
			resize(gray, smallImg, Size(DETECTION_WIDTH, scaleHeight));
		}
		else
		{
			smallImg = gray;
		}

		Mat equalizedImg;
		equalizeHist(smallImg, equalizedImg);

		//object detection
		//int flags = CASCADE_SCALE_IMAGE;
		int flags = CASCADE_DO_ROUGH_SEARCH;
		Size minFeatureSize(30, 30);
		float searchScaleFactor = 1.1f;
		int minNeighbors = 8;
		vector<Rect> faces;
		faceDetector1.detectMultiScale(equalizedImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);
		int check = 0;
		if (faces.size() <= 0)
		{
			faceDetector2.detectMultiScale(equalizedImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);
			if (faces.size() <= 0) {
				Mat revImg;
				equalizedImg.copyTo(revImg);
				flip(equalizedImg, revImg, 1);
				faceDetector2.detectMultiScale(revImg, faces, searchScaleFactor, minNeighbors, flags, minFeatureSize);
				check = 1;
			}
		}
/*
		//fps
		static int lastClock = 0;
		double currentClock = clock();
		double interval = (double)(currentClock - lastClock);
		double fps = (double) CLOCKS_PER_SEC / interval;
		lastClock = currentClock;
		//cout << " fps:" << fps << " " << interval << endl;
*/
		//Draw a rectangle around the object.
		if (cameraFrame.cols > DETECTION_WIDTH)
		{
			if (check <= 0)
			{
				for (int i = 0; i < faces.size(); i++)
				{
					faces[i].x = cvRound(faces[i].x*scale);
					faces[i].y = cvRound(faces[i].y*scale);
					faces[i].width = cvRound(faces[i].width*scale);
					faces[i].height = cvRound(faces[i].height*scale);
					rectangle(cameraFrame, faces[i], Scalar(255, 0, 0), 2);
				}
			}
			else
			{
				for (int i = 0; i < faces.size(); i++)
				{
					faces[i].width = cvRound(faces[i].width*scale);
					faces[i].height = cvRound(faces[i].height*scale);
					faces[i].x = Frame_WIDTH - cvRound(faces[i].x*scale) - faces[i].width;
					faces[i].y = cvRound(faces[i].y*scale);
					rectangle(cameraFrame, faces[i], Scalar(255, 0, 0), 2);
				}
			}
			//Find directions
			if (faces.size())
			{
				midFaceX = faces[0].x + (faces[0].width / 2);
				midFaceY = faces[0].y + (faces[0].height / 2);
				if (midFaceX - Frame_M_WIDTH > Scale_lim)
					Level_p = 1;
				if (midFaceY - Frame_M_HEIGHT > Scale_lim)
					Level_t = 1;

				if (midFaceX - Frame_M_WIDTH < -Scale_lim){
					Level_p = -1;
				}
				if (midFaceY - Frame_M_HEIGHT < -Scale_lim){
					Level_t = -1;
				}

			}
		}
		//Send signal to RPI
		if(faces.size() != 0)
		{
		
			if(abs(midFaceX - Frame_M_WIDTH) <= Scale_lim){
				Level_p =0;
				digitalWrite(Power_pan, 0);
			}
			if(abs(midFaceY - Frame_M_HEIGHT) <= Scale_lim){
				Level_t = 0;
				digitalWrite(Power_tilt, 0);
			}
			digitalWrite(Power_pan, abs(Level_p));
			digitalWrite(Power_tilt, abs(Level_p));
	
			imshow("webcam", cameraFrame);
			if(Level_p<0){
				digitalWrite(Direction_pan, 0);
				digitalWrite(Direction_panl,1);
			}
			else if(Level_p >0)
			{
				digitalWrite(Direction_pan, 1);
				digitalWrite(Direction_panl,0);
			}		
			if(Level_t<0){
	                        digitalWrite(Direction_tilt, 0);
				digitalWrite(Direction_tiltd,1);
	                }
	                else if(Level_t >0)
	                {
	                        digitalWrite(Direction_tilt, 1);
				digitalWrite(Direction_tiltd,0);
        	        }

		//	digitalWrite(Direction, Level);
		//	int power = digitalRead(Power);
		//	int Direc = digitalRead(!Direction);
		//	printf("Direction : %d \n",Direc);
		//	printf("power : %d \n", power);
		//	printf("Level : %d \n",Level );
		}
		//Stop the motor
		else
		{
			digitalWrite(Direction_pan,0);
			digitalWrite(Direction_panl,0);
			digitalWrite(Power_tilt,0);
		}
	if ((char)waitKey(20) == 27) break;
	}
	return 0;
}
