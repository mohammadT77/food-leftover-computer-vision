#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/core.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

// // #include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;

Mat src;
Mat dst;
const char* window_name = "Edge Map";
const char* window_name2 = "original image";



int main( int argc, char** argv )
{

   try {
      if (argc < 2) {
         throw invalid_argument("Not enough arguments!");
      }    
   } catch (const invalid_argument& e) {
      cerr << "Invalid argument: " << e.what() << endl;
      return -1;
   }
   Mat src = imread(argv[1], IMREAD_COLOR);

   if( src.empty() )
   {
   std::cout << "Could not open or find the image!\n" << std::endl;
   std::cout << "Usage: " << argv[0] << " <Input image>" << std::endl;
   return -1;
   }



   // cvtColor( src, src, COLOR_BGR2GRAY );
   // namedWindow( window_name, WINDOW_AUTOSIZE );
   namedWindow( window_name2, WINDOW_AUTOSIZE );
   imshow( window_name2, src );
   blur( src, src, Size(5,5) );
   
   cv::Mat imageRGB;
   cv::cvtColor(src, imageRGB, cv::COLOR_BGR2RGB);

   // Reshape the image to a 2D matrix of pixels
   cv::Mat reshapedImage = imageRGB.reshape(1, imageRGB.rows * imageRGB.cols);

   cout << src.size()<<endl;
   // cout << reshapedImage<<endl;

   // Convert the reshaped image to float
   cv::Mat reshapedImageFloat;
   reshapedImage.convertTo(reshapedImageFloat, CV_32F);

   // Define the number of clusters
   int numClusters = 5;

   // Set the criteria for k-means algorithm
   cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 1.0);

   // Perform k-means clustering
   cv::Mat labels, centers;
   cv::kmeans(reshapedImageFloat, numClusters, labels, criteria, 1, cv::KMEANS_RANDOM_CENTERS, centers);

   // Reshape the labels to the original image size
   cv::Mat segmented = labels.reshape(1, src.rows);


   // Convert the labels to 8-bit image
   cv::Mat segmented8Bit;
   segmented.convertTo(segmented8Bit, CV_8U);

   for(int i=0; i<segmented8Bit.rows; i++){
      for(int j=0; j < segmented8Bit.cols; j++){
         switch (segmented8Bit.at<uchar>(i,j)){
            case 0: segmented8Bit.at<uchar>(i,j) = 0; break;
            case 1: segmented8Bit.at<uchar>(i,j) = 50; break;
            case 2: segmented8Bit.at<uchar>(i,j) = 100; break;
            case 3: segmented8Bit.at<uchar>(i,j) = 150; break;
            case 4: segmented8Bit.at<uchar>(i,j) = 200; break;
            default: segmented8Bit.at<uchar>(i,j) = 255;
         }
      }
   }

   // Display the segmented image
   cv::imshow("Segmented Image", segmented8Bit);
   cv::waitKey(0);
   return 0;
}