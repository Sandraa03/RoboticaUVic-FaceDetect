#include <iostream>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

bool flag;

int main(int argc, char *argv[]) 
{
    //OpenCV video capture object
    cv::VideoCapture camera;

    //OpenCV image object
    cv::Mat image;
    cv::Mat image2;

    //camera id
    int cam_id;
    flag = true;

    //check user args
    switch(argc)
    {
      case 1: //no argument provided, try /dev/video0
        cam_id = 0;
        break;
      case 2: //an argument is provided. Get it and set cam_id
        cam_id = atoi(argv[1]);
        break;
      default:
        std::cout << "Invalid number of arguments." << std::endl;
        std::cout << "EXIT program." << std::endl;
        break;
    }

    //advertising to the user
    std::cout << "Opening video device " << cam_id << std::endl;

    //open the video stream and make sure it's opened
    if( !camera.open(cam_id) )
    {
      std::cout << "Error opening the camera. Invalid device id. EXIT program." << std::endl;
      return -1;
    }

    //Initialize face detector object
    cv::CascadeClassifier face_cascade;
    face_cascade.load("../haarcascade_frontalface_default.xml");


  //Main loop
  while (flag)
  {
    //Load next frame  
    if(!camera.read(image))
    {
      std::cout << "No frame" << std::endl;
      cv::waitKey();
      flag = false;
    }

  //Convert frame to grayscale
  cv::Mat gray;
  cv::cvtColor(image, gray, CV_BGR2GRAY);

  //Detect faces as rectangles
  vector<Rect> faces;
  face_cascade.detectMultiScale(gray, faces, 1.3, 4, CASCADE_SCALE_IMAGE, Size(30, 30));

  //Clone image
  image2 = image.clone();

  //Draw the rectangle in duplicated image for each detected face
  for (int i = 0; i < faces.size(); i++)
  {
    Rect r = faces[i];       
    rectangle(image2, Point(r.x+r.width, r.y+r.height), Point(r.x, r.y), Scalar(0, 255, 0), 2);
  }

  //show in window
  cv::imshow("faces", image2);

  if(cv::waitKey(1) >= 0) break;

  }

  destroyAllWindows();
}