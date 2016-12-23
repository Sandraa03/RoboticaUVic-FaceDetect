#include <iostream>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>
#include "cv.h"
#include "highgui.h"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

bool flag;

int main(int argc, char *argv[]) 
{
    //OpenCV video capture object
    cv::VideoCapture camera;

    //OpenCV image object
    cv::Mat image;
    cv::Mat image2;  //Cloned image

    int cam_id;
    flag = true;

    //Load Hat and Moustache pictures
    Mat hat = imread("../img/hat.png", -1); //load hat
    Mat hat_resized; //hat image once resized
    Mat moustache = imread("../img/moustache.png", -1); //load moustache 
    Mat moustache_resized; //moustache image once resized

    //3 variables to get pixels combination on hat and moustache fussion
    double color_pixel_0, color_pixel_1, color_pixel_2;
    
    if(!hat.data ) // Check for invalid input image 
    {
        std::cout <<  "Could not open or find the image hat.png" << std::endl ;
        return -1;
    }
    if(! moustache.data ) // Check for invalid input image 
    {
        std::cout <<  "Could not open or find the image moustache.png" << std::endl ;
        return -1;
    }


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

    //Do ... for each detected face
    for (int i = 0; i < faces.size(); i++)
    {
      Rect face_i = faces[i];
      //Add hat and moustache to picture
      int facew = face_i.width; //face width
      int faceh = face_i.height; //face height
      Size hat_size(facew,faceh); //resize hat picture giving same face width
      resize(hat, hat_resized, hat_size);

      Size moustache_size(facew/2,faceh/2); //resize moustache picture giving half face width
      resize(moustache, moustache_resized, moustache_size);

      double hat_locate = 0.50; //Variable to move up hat from face position
      double moustache_locate_y = 0.50; //Variable to move down moustache from face position
      double moustache_move_x = (facew - moustache_resized.size[0])/2; //Variable to move right moustache from face position

      //Overlay hat and moustache
      for ( int j = 0; j < faceh ; j++)
      {
        for ( int k = 0; k < facew; k++)
        {

          //determine the opacity of the foregrond pixel, using its fourth (alpha) channel for pictures picture.
          double alpha_hat = hat_resized.at<cv::Vec4b>(j, k)[3] / 255.0;
          color_pixel_0 = (hat_resized.at<cv::Vec4b>(j, k)[0] * (alpha_hat)) + ((image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[0])* (1.0-alpha_hat));
          color_pixel_1 = (hat_resized.at<cv::Vec4b>(j, k)[1] * (alpha_hat)) + ((image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[1])* (1.0-alpha_hat));
          color_pixel_2 = (hat_resized.at<cv::Vec4b>(j, k)[2] * (alpha_hat)) + ((image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[2])* (1.0-alpha_hat));

          if((face_i.y +j-(faceh*hat_locate))>0)
          {
            image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[0] = color_pixel_0 ;
            image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[1] = color_pixel_1 ;
            image2.at<cv::Vec3b>((face_i.y +j-(faceh*hat_locate)), (face_i.x +k))[2] = color_pixel_2 ;
          }

          if((j<(faceh/2))&&(k<(facew/2)))
          {
            //determine the opacity of the foregrond pixel, using its fourth (alpha) channel for moustache picture.
            double alpha_moustache = moustache_resized.at<cv::Vec4b>(j, k)[3] / 255.0;
            color_pixel_0 = (moustache_resized.at<cv::Vec4b>(j, k)[0] * (alpha_moustache)) + ((image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[0])* (1.0-alpha_moustache));
            color_pixel_1 = (moustache_resized.at<cv::Vec4b>(j, k)[1] * (alpha_moustache)) + ((image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[1])* (1.0-alpha_moustache));
            color_pixel_2 = (moustache_resized.at<cv::Vec4b>(j, k)[2] * (alpha_moustache)) + ((image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[2])* (1.0-alpha_moustache));

            image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[0] = color_pixel_0 ;
            image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[1] = color_pixel_1 ;
            image2.at<cv::Vec3b>((face_i.y +j+(faceh*moustache_locate_y)), (face_i.x +k+(moustache_move_x)))[2] = color_pixel_2 ;
          }
        }
      }
    }

    //show in window
    cv::imshow("faces", image2);

    if(cv::waitKey(1) >= 0) break;

    }

    destroyAllWindows();
  }