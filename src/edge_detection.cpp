#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;

Mat src, src_gray;
Mat detected_edges;
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



   cvtColor( src, src_gray, COLOR_BGR2GRAY );
   namedWindow( window_name, WINDOW_AUTOSIZE );
   namedWindow( window_name2, WINDOW_AUTOSIZE );
   imshow( window_name2, src );
   blur( src_gray, src_gray, Size(5,5) );
   Canny(src_gray,  detected_edges, 30, 50, 3 );

   imshow( window_name, detected_edges );
   waitKey(0);
   // Find contours
   std::vector<std::vector<Point>> contours;
   std::vector<Vec4i> hierarchy;
   findContours(detected_edges, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
 

   // Iterate over the contours
   for (auto it = contours.begin(); it != contours.end();) {
      // Calculate the area of the current contour
      double area = cv::contourArea(*it);

      // Check if the area is less than 30
      if (area < 30) {
         // Erase the contour if it doesn't meet the criteria
         it = contours.erase(it);
      } else {
         // Move to the next contour
         ++it;
      }
   }

   std::cout<< contours.size()<<std::endl;
   std::cout<< hierarchy.size();
   // Draw contours
   Mat contour_image = Mat::zeros(detected_edges.size(), CV_8UC3);
   for (size_t i = 0; i < contours.size(); i++)
   {
      drawContours(contour_image, contours, static_cast<int>(i), Scalar(0, 0, 255), 1);
   }



   imshow(window_name, detected_edges);
   imshow("Contours", contour_image);

   waitKey(0);

   return 0;
}